// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Aten op mappers for normalization, embedding, and loss function overloads (Wave 12).
//!
//! Adds additional overloads commonly produced by `torch.export` for ops whose
//! `.default` variants were mapped in earlier waves:
//!
//! - Normalization: `_native_batch_norm_legit`, `_native_batch_norm_legit.no_stats`,
//!   `cudnn_batch_norm`, `instance_norm` with affine params, `layer_norm` with
//!   normalized_shape list, `group_norm` with optional weight/bias
//! - Embedding: `embedding_bag`, `embedding` with padding_idx
//! - Loss: `cross_entropy_loss` with label_smoothing + ignore_index,
//!   `nll_loss_nd`, `nll_loss2d_forward`, `binary_cross_entropy` with weight,
//!   `mse_loss` backward, `l1_loss` backward, `smooth_l1_loss` backward,
//!   `kl_div` backward

use nn_core::dyn_tensor::trace::TraceOp;

use super::{
    first_tensor_name, get_arg, optional_bool, optional_float, optional_int, optional_weight,
    require_int, require_tensor_name, resolve_weight, safe_usize, ImportError, Node, OpMapContext,
};

// =========================================================================
// Batch normalization: _native_batch_norm_legit (training variant)
// =========================================================================

/// Map `aten._native_batch_norm_legit.default` to `TraceOp::BatchNorm`.
///
/// torch.export signature:
///   `(input, weight?, bias?, running_mean?, running_var?, training, momentum, eps)`
///
/// This is the training-mode batch norm emitted by newer PyTorch export paths.
/// During inference (training=false), it behaves identically to the no_training
/// variant already mapped. We ignore the `training` and `momentum` args since
/// nn uses running stats for inference.
pub(super) fn map_native_batch_norm_legit(
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

/// Map `aten._native_batch_norm_legit.no_stats` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(input, weight?, bias?, training, momentum, eps)`
///
/// Variant without running statistics — uses batch statistics instead.
/// Common in training graphs where running_mean/running_var are not maintained.
/// Falls back to Custom since `TraceOp::BatchNorm` requires running stats.
pub(super) fn map_native_batch_norm_legit_no_stats(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);
    let training = optional_bool(node, "training", true);
    Ok((
        TraceOp::Custom {
            name: format!("batch_norm_no_stats_eps{eps}_train{training}"),
        },
        vec![input],
    ))
}

/// Map `aten.cudnn_batch_norm.default` to `TraceOp::BatchNorm`.
///
/// torch.export signature:
///   `(input, weight, bias, running_mean, running_var, training, momentum, eps)`
///
/// cuDNN-dispatched batch norm. Semantically identical to native_batch_norm
/// for inference. Some PyTorch export traces from CUDA models emit this target.
pub(super) fn map_cudnn_batch_norm(
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

// =========================================================================
// Layer normalization: normalized_shape list variant
// =========================================================================

/// Map `aten.layer_norm.default` with optional None weight/bias.
///
/// Some torch.export traces emit layer_norm where weight and/or bias are
/// `None` (unaffined layer norm). The existing mapper in `op_map_impl.rs`
/// requires both. This variant handles the case by creating zero-sized
/// weight refs when None.
pub(super) fn map_layer_norm_optional_affine(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);

    // Weight and bias may be None for unaffined layer norm.
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
            // Unaffined layer norm: no learned scale/shift.
            Ok((
                TraceOp::Custom {
                    name: format!("layer_norm_no_affine_eps{eps}"),
                },
                vec![input],
            ))
        }
    }
}

// =========================================================================
// Group normalization: optional weight/bias variant
// =========================================================================

/// Map `aten.group_norm.default` with optional None weight/bias.
///
/// Some torch.export traces emit group_norm where weight and/or bias are
/// `None`. The existing mapper requires both. This variant falls back to
/// a Custom op when affine parameters are missing.
pub(super) fn map_group_norm_optional_affine(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = require_tensor_name(node, "input")?;
    let num_groups = safe_usize(require_int(node, "num_groups")?, "num_groups", &node.target)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);

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
        _ => Ok((
            TraceOp::Custom {
                name: format!("group_norm_no_affine_g{num_groups}_eps{eps}"),
            },
            vec![input],
        )),
    }
}

// =========================================================================
// Instance normalization: affine variant with weight/bias/running stats
// =========================================================================

/// Map `aten.instance_norm.default` with affine parameters.
///
/// torch.export signature:
///   `(input, weight?, bias?, running_mean?, running_var?, use_input_stats, momentum, eps)`
///
/// The existing mapper only extracts eps. This variant handles the full
/// signature including optional weight/bias for affine instance norm and
/// optional running statistics.
pub(super) fn map_instance_norm_affine(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let eps = optional_float(node, "eps").unwrap_or(1e-5);

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

    let has_affine = weight_name.is_some() && bias_name.is_some();
    let weight = optional_weight(weight_name.as_deref(), ctx);
    let bias = optional_weight(bias_name.as_deref(), ctx);

    if has_affine {
        if let (Some(_w), Some(_b)) = (weight, bias) {
            return Ok((
                TraceOp::Custom {
                    name: format!("instance_norm_affine_eps{eps}"),
                },
                vec![input],
            ));
        }
    }

    Ok((TraceOp::InstanceNorm { eps }, vec![input]))
}

// =========================================================================
// Embedding: padding_idx and embedding_bag variants
// =========================================================================

/// Map `aten.embedding.padding_idx` to `TraceOp::Embedding`.
///
/// torch.export signature: `(weight, indices, padding_idx, scale_grad_by_freq, sparse)`
///
/// Same as `embedding.default` but with explicit `padding_idx`. We ignore
/// `padding_idx` for inference (it only affects gradient computation) and
/// `sparse`/`scale_grad_by_freq` which are also training-only.
pub(super) fn map_embedding_padding_idx(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let weight_name = require_tensor_name(node, "weight")?;
    let indices = require_tensor_name(node, "indices")?;
    let weight = resolve_weight(&weight_name, ctx)?;
    // padding_idx, sparse, scale_grad_by_freq are training-only; ignored for inference.
    Ok((TraceOp::Embedding { weight }, vec![indices]))
}

/// Map `aten._embedding_bag.default` / `aten.embedding_bag.default` to `TraceOp::Custom`.
///
/// torch.export signature:
///   `(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights?, ...)`
///
/// Embedding bag performs a reduction (sum/mean/max) over bags of embeddings.
/// `mode`: 0=sum, 1=mean, 2=max.
pub(super) fn map_embedding_bag(
    node: &Node,
    ctx: &OpMapContext<'_>,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let weight_name = require_tensor_name(node, "weight")?;
    let indices = require_tensor_name(node, "indices")?;
    let offsets = require_tensor_name(node, "offsets")?;
    let mode = optional_int(node, "mode").unwrap_or(0); // 0=sum, 1=mean, 2=max
    let _weight = resolve_weight(&weight_name, ctx)?;
    Ok((
        TraceOp::Custom {
            name: format!("embedding_bag_mode{mode}"),
        },
        vec![indices, offsets],
    ))
}

// =========================================================================
// Cross-entropy loss: full parameter extraction
// =========================================================================

/// Map `aten.cross_entropy_loss.default` with full parameter extraction.
///
/// torch.export signature:
///   `(self, target, weight?, reduction, ignore_index, label_smoothing)`
///
/// Extends the wave 9 mapper by extracting `ignore_index` and `label_smoothing`
/// into the custom op name for downstream processing.
pub(super) fn map_cross_entropy_loss_full(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    let ignore_index = optional_int(node, "ignore_index").unwrap_or(-100);
    let label_smoothing = optional_float(node, "label_smoothing").unwrap_or(0.0);
    Ok((
        TraceOp::Custom {
            name: format!("cross_entropy_loss_r{reduction}_ig{ignore_index}_ls{label_smoothing}"),
        },
        vec![input, target],
    ))
}

// =========================================================================
// NLL loss: N-dim and 2D variants
// =========================================================================

/// Map `aten.nll_loss_nd.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, target, weight?, reduction, ignore_index)`
///
/// N-dimensional NLL loss. Handles arbitrary input dimensions (not just 2D).
/// Some export paths produce `nll_loss_nd` instead of `nll_loss`.
pub(super) fn map_nll_loss_nd(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    let ignore_index = optional_int(node, "ignore_index").unwrap_or(-100);
    Ok((
        TraceOp::Custom {
            name: format!("nll_loss_nd_r{reduction}_ig{ignore_index}"),
        },
        vec![input, target],
    ))
}

/// Map `aten.nll_loss2d_forward.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(self, target, weight?, reduction, ignore_index)`
///
/// 2D spatial NLL loss, used in image segmentation models.
/// The `forward` variant outputs (loss, total_weight) tuple.
pub(super) fn map_nll_loss2d_forward(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    let ignore_index = optional_int(node, "ignore_index").unwrap_or(-100);
    Ok((
        TraceOp::Custom {
            name: format!("nll_loss2d_forward_r{reduction}_ig{ignore_index}"),
        },
        vec![input, target],
    ))
}

// =========================================================================
// Binary cross-entropy: weight tensor variant
// =========================================================================

/// Map `aten.binary_cross_entropy.default` with weight argument.
///
/// torch.export signature: `(self, target, weight?, reduction)`
///
/// Extends the wave 7 mapper by tracking whether a weight tensor is provided.
/// Weight is per-element scaling applied to the loss before reduction.
pub(super) fn map_binary_cross_entropy_weighted(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let input = first_tensor_name(node)?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    let has_weight = get_arg(node, "weight")
        .ok()
        .is_some_and(|a| !a.is_none() && a.as_tensor_name().is_some());
    let mut inputs = vec![input, target];
    if has_weight {
        if let Ok(w) = require_tensor_name(node, "weight") {
            inputs.push(w);
        }
    }
    Ok((
        TraceOp::Custom {
            name: format!(
                "binary_cross_entropy_r{reduction}_w{}",
                if has_weight { "yes" } else { "no" }
            ),
        },
        inputs,
    ))
}

// =========================================================================
// MSE loss: backward variant (training graph exports)
// =========================================================================

/// Map `aten.mse_loss_backward.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(grad_output, self, target, reduction)`
///
/// Backward pass of MSE loss. Appears in exported training graphs.
pub(super) fn map_mse_loss_backward(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let grad_output = require_tensor_name(node, "grad_output")?;
    let self_input = require_tensor_name(node, "self")?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    Ok((
        TraceOp::Custom {
            name: format!("mse_loss_backward_r{reduction}"),
        },
        vec![grad_output, self_input, target],
    ))
}

// =========================================================================
// L1 loss: backward variant (training graph exports)
// =========================================================================

/// Map `aten.l1_loss_backward.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(grad_output, self, target, reduction)`
///
/// Backward pass of L1 loss. Appears in exported training graphs.
pub(super) fn map_l1_loss_backward(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let grad_output = require_tensor_name(node, "grad_output")?;
    let self_input = require_tensor_name(node, "self")?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    Ok((
        TraceOp::Custom {
            name: format!("l1_loss_backward_r{reduction}"),
        },
        vec![grad_output, self_input, target],
    ))
}

// =========================================================================
// Smooth L1 loss: backward variant (training graph exports)
// =========================================================================

/// Map `aten.smooth_l1_loss_backward.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(grad_output, self, target, reduction, beta)`
///
/// Backward pass of smooth L1 (Huber) loss. Appears in exported training graphs.
pub(super) fn map_smooth_l1_loss_backward(
    node: &Node,
) -> Result<(TraceOp, Vec<String>), ImportError> {
    let grad_output = require_tensor_name(node, "grad_output")?;
    let self_input = require_tensor_name(node, "self")?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    let beta = optional_float(node, "beta").unwrap_or(1.0);
    Ok((
        TraceOp::Custom {
            name: format!("smooth_l1_loss_backward_r{reduction}_b{beta}"),
        },
        vec![grad_output, self_input, target],
    ))
}

// =========================================================================
// KL divergence: backward variant (training graph exports)
// =========================================================================

/// Map `aten.kl_div_backward.default` to `TraceOp::Custom`.
///
/// torch.export signature: `(grad_output, self, target, reduction, log_target)`
///
/// Backward pass of KL divergence. Appears in exported training graphs.
pub(super) fn map_kl_div_backward(node: &Node) -> Result<(TraceOp, Vec<String>), ImportError> {
    let grad_output = require_tensor_name(node, "grad_output")?;
    let self_input = require_tensor_name(node, "self")?;
    let target = require_tensor_name(node, "target")?;
    let reduction = optional_int(node, "reduction").unwrap_or(1);
    let log_target = optional_bool(node, "log_target", false);
    Ok((
        TraceOp::Custom {
            name: format!("kl_div_backward_r{reduction}_lt{log_target}"),
        },
        vec![grad_output, self_input, target],
    ))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
#[path = "op_map_impl_wave12_tests.rs"]
mod tests;

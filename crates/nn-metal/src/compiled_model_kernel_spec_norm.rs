// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Norm-family `spec_*()` builder functions.
//!
//! Covers InstanceNorm, LayerNorm, AddLayerNorm, ChannelsFirstLayerNorm,
//! and AdaLayerNorm. Each builds a [`KernelSpec`] capturing kernel name,
//! MSL source, grid/threadgroup dimensions, and buffer bindings.
//!
//! Split from `compiled_model_kernel_spec_builders.rs` per 500-line limit
//! and design doc D5.
//!
//! Part of #3503 (KernelSpec unification).

use super::{KernelBinding, KernelSpec, SpecDispatchMode};

/// Threadgroup size for fused norm kernels (InstanceNorm, LayerNorm, AddLayerNorm,
/// ChannelsFirstLayerNorm, AdaLayerNorm, AdainSnake, AdainLeakyRelu).
/// Matches the TG_SIZE constants in each `dyn_tensor_metal_*_fused.rs`.
pub(crate) const NORM_TG_SIZE: u32 = 256;

/// Build a [`KernelSpec`] for fused InstanceNorm.
///
/// Extracts the kernel parameter logic from both the DynTensor bridge path
/// (`gpu_instance_norm_fused`) and the NativeOp executor, capturing:
/// - Kernel name: `fused_instance_norm_{scalar_type}`
/// - MSL source: from `fused_instance_norm_msl`
/// - Grid: one threadgroup per `batch * channels` row
/// - Bindings: input (Edge 0), output (Output), spatial (Constant u32), eps (Constant f32)
///
/// Part of #3503 D2.
pub(crate) fn spec_instance_norm(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
) -> Result<KernelSpec, String> {
    if input_shape.len() < 3 {
        return Err(format!(
            "spec_instance_norm: need rank >= 3, got {}",
            input_shape.len()
        ));
    }

    let batch = input_shape[0];
    let channels = input_shape[1];
    let spatial: usize = input_shape[2..].iter().product();

    let flat_rows = batch.checked_mul(channels).ok_or_else(|| {
        format!("spec_instance_norm: batch*channels overflow ({batch} * {channels})")
    })?;
    let total_elems = flat_rows.checked_mul(spatial).ok_or_else(|| {
        format!("spec_instance_norm: total overflow ({flat_rows} * {spatial})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_instance_norm_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::instance_norm_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::instance_norm_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_instance_norm: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_instance_norm: flat_rows {flat_rows} exceeds u32"))?;
    let spatial_u32 = u32::try_from(spatial)
        .map_err(|_| format!("spec_instance_norm: spatial {spatial} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_instance_norm: output bytes overflow ({total_elems} * {elem_bytes})")
    })?;

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [flat_rows_u32, 1, 1],
        threadgroup: [NORM_TG_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: 0,
        output_bytes,
        bindings: vec![
            (0, KernelBinding::Edge(0)),
            (1, KernelBinding::Output),
            (2, KernelBinding::constant_u32(spatial_u32)),
            (3, KernelBinding::constant_f32(eps)),
        ],
        param_count: 1,
        fast_math: false,
    })
}

/// Build a [`KernelSpec`] for fused LayerNorm.
///
/// Single Metal dispatch: `(x - mean) / sqrt(var + eps) * weight + bias`.
/// One threadgroup per row (flat_rows = product of dims except last).
///
/// Buffer layout:
///   0: input `[..rows, hidden_dim]` (Edge 0)
///   1: weight `[hidden_dim]` (Weight "weight")
///   2: bias `[hidden_dim]` (Weight "bias")
///   3: output (Output)
///   4: hidden_dim (Constant u32)
///   5: eps (Constant f32)
///
/// Part of #3503 D3.
pub(crate) fn spec_layer_norm(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
) -> Result<KernelSpec, String> {
    if input_shape.len() < 2 {
        return Err(format!(
            "spec_layer_norm: need rank >= 2, got {}",
            input_shape.len()
        ));
    }

    let flat_rows: usize = input_shape[..input_shape.len() - 1].iter().product();
    let total_elems = flat_rows.checked_mul(hidden_dim).ok_or_else(|| {
        format!("spec_layer_norm: total overflow ({flat_rows} * {hidden_dim})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_layer_norm_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::layer_norm_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::layer_norm_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_layer_norm: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_layer_norm: flat_rows {flat_rows} exceeds u32"))?;
    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| format!("spec_layer_norm: hidden_dim {hidden_dim} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_layer_norm: output bytes overflow ({total_elems} * {elem_bytes})")
    })?;

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [flat_rows_u32, 1, 1],
        threadgroup: [NORM_TG_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: 0,
        output_bytes,
        bindings: vec![
            (0, KernelBinding::Edge(0)),
            (1, KernelBinding::Weight("weight".into())),
            (2, KernelBinding::Weight("bias".into())),
            (3, KernelBinding::Output),
            (4, KernelBinding::constant_u32(hidden_dim_u32)),
            (5, KernelBinding::constant_f32(eps)),
        ],
        param_count: 3,
        fast_math: false,
    })
}

/// Build a [`KernelSpec`] for fused Add + LayerNorm.
///
/// Single Metal dispatch: `LayerNorm(a + b, weight, bias, eps)`.
/// Two edge inputs (residual a, new value b), one threadgroup per row.
///
/// Buffer layout:
///   0: a `[..rows, hidden_dim]` (Edge 0)
///   1: b `[..rows, hidden_dim]` (Edge 1)
///   2: weight `[hidden_dim]` (Weight "weight")
///   3: bias `[hidden_dim]` (Weight "bias")
///   4: output (Output)
///   5: hidden_dim (Constant u32)
///   6: eps (Constant f32)
///
/// Part of #3503 D3.
pub(crate) fn spec_add_layer_norm(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
) -> Result<KernelSpec, String> {
    if input_shape.len() < 2 {
        return Err(format!(
            "spec_add_layer_norm: need rank >= 2, got {}",
            input_shape.len()
        ));
    }

    let flat_rows: usize = input_shape[..input_shape.len() - 1].iter().product();
    let total_elems = flat_rows.checked_mul(hidden_dim).ok_or_else(|| {
        format!("spec_add_layer_norm: total overflow ({flat_rows} * {hidden_dim})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_add_layer_norm_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::add_layer_norm_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::add_layer_norm_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_add_layer_norm: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_add_layer_norm: flat_rows {flat_rows} exceeds u32"))?;
    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| format!("spec_add_layer_norm: hidden_dim {hidden_dim} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_add_layer_norm: output bytes overflow ({total_elems} * {elem_bytes})")
    })?;

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [flat_rows_u32, 1, 1],
        threadgroup: [NORM_TG_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: 0,
        output_bytes,
        bindings: vec![
            (0, KernelBinding::Edge(0)),
            (1, KernelBinding::Edge(1)),
            (2, KernelBinding::Weight("weight".into())),
            (3, KernelBinding::Weight("bias".into())),
            (4, KernelBinding::Output),
            (5, KernelBinding::constant_u32(hidden_dim_u32)),
            (6, KernelBinding::constant_f32(eps)),
        ],
        param_count: 4,
        fast_math: false,
    })
}

/// Build a [`KernelSpec`] for ChannelsFirstLayerNorm.
///
/// Normalizes over dim 1 (channel dimension) of a `[B, C, T]` tensor.
/// One threadgroup per `(b, t)` pair = B*T threadgroups.
///
/// Without LeakyReLU buffer layout:
///   0: input `[B, C, T]` (Edge 0)
///   1: weight `[C]` (Weight "weight")
///   2: bias `[C]` (Weight "bias")
///   3: output (Output)
///   4: channels (Constant u32)
///   5: time_steps (Constant u32)
///   6: eps (Constant f32)
///
/// With LeakyReLU, adds buffer 7: slope (Constant f32).
///
/// Part of #3503 D3.
pub(crate) fn spec_channels_first_layer_norm(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
    channels: usize,
    leaky_relu_slope: Option<f32>,
) -> Result<KernelSpec, String> {
    if input_shape.len() != 3 {
        return Err(format!(
            "spec_channels_first_layer_norm: need rank == 3, got {}",
            input_shape.len()
        ));
    }

    let batch = input_shape[0];
    let time_steps = input_shape[2];
    let bt = batch.checked_mul(time_steps).ok_or_else(|| {
        format!("spec_channels_first_layer_norm: B*T overflow ({batch} * {time_steps})")
    })?;
    let total_elems = bt.checked_mul(channels).ok_or_else(|| {
        format!("spec_channels_first_layer_norm: total overflow ({bt} * {channels})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let (kernel_name, msl_source) = if let Some(slope) = leaky_relu_slope {
        let _ = slope; // used in binding below
        let name = format!("fused_channels_first_ln_leaky_relu_{scalar_str}");
        let msl = match scalar_type {
            nn_dsl::ir::ScalarType::F32 => {
                crate::dyn_tensor_metal::channels_first_ln_leaky_relu_msl_source()
            }
            nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
                crate::dyn_tensor_metal::channels_first_ln_leaky_relu_f16_msl_source()
            }
            _ => {
                return Err(format!(
                    "spec_channels_first_layer_norm: unsupported scalar type {scalar_type:?}"
                ))
            }
        };
        (name, msl)
    } else {
        let name = format!("fused_channels_first_layer_norm_{scalar_str}");
        let msl = match scalar_type {
            nn_dsl::ir::ScalarType::F32 => {
                crate::dyn_tensor_metal::channels_first_layer_norm_msl_source()
            }
            nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
                crate::dyn_tensor_metal::channels_first_layer_norm_f16_msl_source()
            }
            _ => {
                return Err(format!(
                    "spec_channels_first_layer_norm: unsupported scalar type {scalar_type:?}"
                ))
            }
        };
        (name, msl)
    };

    let bt_u32 = u32::try_from(bt)
        .map_err(|_| format!("spec_channels_first_layer_norm: B*T {bt} exceeds u32"))?;
    let channels_u32 = u32::try_from(channels)
        .map_err(|_| format!("spec_channels_first_layer_norm: channels {channels} exceeds u32"))?;
    let time_steps_u32 = u32::try_from(time_steps)
        .map_err(|_| {
            format!("spec_channels_first_layer_norm: time_steps {time_steps} exceeds u32")
        })?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!(
            "spec_channels_first_layer_norm: output bytes overflow ({total_elems} * {elem_bytes})"
        )
    })?;

    let mut bindings = vec![
        (0, KernelBinding::Edge(0)),
        (1, KernelBinding::Weight("weight".into())),
        (2, KernelBinding::Weight("bias".into())),
        (3, KernelBinding::Output),
        (4, KernelBinding::constant_u32(channels_u32)),
        (5, KernelBinding::constant_u32(time_steps_u32)),
        (6, KernelBinding::constant_f32(eps)),
    ];

    if let Some(slope) = leaky_relu_slope {
        bindings.push((7, KernelBinding::constant_f32(slope)));
    }

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [bt_u32, 1, 1],
        threadgroup: [NORM_TG_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: 0,
        output_bytes,
        bindings,
        param_count: 3,
        fast_math: false,
    })
}

/// Build a [`KernelSpec`] for fused AdaLayerNorm.
///
/// Single Metal dispatch: LayerNorm(x, w, b) -> (1+gamma)*normed+beta.
/// One threadgroup per row (flat_rows = batch * time_steps).
///
/// Buffer layout:
///   0: x `[B, T, C]` (Edge 0)
///   1: gamma `[B, 1, C]` (Edge 1)
///   2: beta `[B, 1, C]` (Edge 2)
///   3: norm_weight `[C]` (Weight "norm_weight")
///   4: norm_bias `[C]` (Weight "norm_bias")
///   5: output (Output)
///   6: hidden_dim (Constant u32)
///   7: time_steps (Constant u32)
///   8: eps (Constant f32)
///
/// Part of #3503 D3.
pub(crate) fn spec_ada_layer_norm(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
) -> Result<KernelSpec, String> {
    if input_shape.len() < 3 {
        return Err(format!(
            "spec_ada_layer_norm: need rank >= 3, got {}",
            input_shape.len()
        ));
    }

    let batch = input_shape[0];
    // Middle dimensions are time/spatial.
    let mid_dims: usize = input_shape[1..input_shape.len() - 1].iter().product();
    let flat_rows = batch.checked_mul(mid_dims).ok_or_else(|| {
        format!("spec_ada_layer_norm: B*mid overflow ({batch} * {mid_dims})")
    })?;
    let total_elems = flat_rows.checked_mul(hidden_dim).ok_or_else(|| {
        format!("spec_ada_layer_norm: total overflow ({flat_rows} * {hidden_dim})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_ada_layer_norm_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::ada_layer_norm_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::ada_layer_norm_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_ada_layer_norm: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_ada_layer_norm: flat_rows {flat_rows} exceeds u32"))?;
    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| format!("spec_ada_layer_norm: hidden_dim {hidden_dim} exceeds u32"))?;
    let time_steps_u32 = u32::try_from(mid_dims)
        .map_err(|_| format!("spec_ada_layer_norm: time_steps {mid_dims} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!(
            "spec_ada_layer_norm: output bytes overflow ({total_elems} * {elem_bytes})"
        )
    })?;

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [flat_rows_u32, 1, 1],
        threadgroup: [NORM_TG_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: 0,
        output_bytes,
        bindings: vec![
            (0, KernelBinding::Edge(0)),
            (1, KernelBinding::Edge(1)),
            (2, KernelBinding::Edge(2)),
            (3, KernelBinding::Weight("norm_weight".into())),
            (4, KernelBinding::Weight("norm_bias".into())),
            (5, KernelBinding::Output),
            (6, KernelBinding::constant_u32(hidden_dim_u32)),
            (7, KernelBinding::constant_u32(time_steps_u32)),
            (8, KernelBinding::constant_f32(eps)),
        ],
        param_count: 5,
        fast_math: false,
    })
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Single-dispatch `spec_*()` builders for GroupNorm, RmsNorm, Snake, and
//! FlashAttention.
//!
//! These are single Metal dispatch kernels with pre-existing MSL source.
//! Each builder follows the same pattern as the norm/fused builders: compute
//! grid/threadgroup/output dimensions, build a binding list, return a
//! `KernelSpec`.
//!
//! Part of #3503 D3 (KernelSpec unification).

use super::{KernelBinding, KernelSpec, SpecDispatchMode};
use super::norm::NORM_TG_SIZE;

// -------------------------------------------------------------------------
// GroupNorm
// -------------------------------------------------------------------------

/// Build a [`KernelSpec`] for fused GroupNorm.
///
/// Single Metal dispatch: `(x - mean) / sqrt(var + eps) * weight + bias`
/// with channel grouping. One threadgroup per `batch * num_groups` group row.
///
/// Buffer layout:
///   0: x `[B, C, *spatial]` (Edge 0)
///   1: weight `[C]` (Weight "weight")
///   2: bias `[C]` (Weight "bias")
///   3: output (Output)
///   4: flat_cols = (C/G)*spatial (Constant u32)
///   5: eps (Constant f32)
///   6: channels_per_group = C/G (Constant u32)
///   7: spatial (Constant u32)
///   8: num_groups (Constant u32)
///
/// Part of #3503 D3.
pub(crate) fn spec_group_norm(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
    num_groups: usize,
) -> Result<KernelSpec, String> {
    if input_shape.len() < 2 {
        return Err(format!(
            "spec_group_norm: need rank >= 2, got {}",
            input_shape.len()
        ));
    }

    let batch = input_shape[0];
    let channels = input_shape[1];
    if num_groups == 0 || !channels.is_multiple_of(num_groups) {
        return Err(format!(
            "spec_group_norm: channels {channels} not divisible by num_groups {num_groups}"
        ));
    }
    let channels_per_group = channels / num_groups;
    let spatial: usize = input_shape[2..].iter().product::<usize>().max(1);

    let flat_rows = batch.checked_mul(num_groups).ok_or_else(|| {
        format!("spec_group_norm: B*G overflow ({batch} * {num_groups})")
    })?;
    let flat_cols = channels_per_group.checked_mul(spatial).ok_or_else(|| {
        format!("spec_group_norm: (C/G)*spatial overflow ({channels_per_group} * {spatial})")
    })?;
    let total_elems = flat_rows.checked_mul(flat_cols).ok_or_else(|| {
        format!("spec_group_norm: total overflow ({flat_rows} * {flat_cols})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_group_norm_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::group_norm_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::group_norm_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_group_norm: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_group_norm: flat_rows {flat_rows} exceeds u32"))?;
    let flat_cols_u32 = u32::try_from(flat_cols)
        .map_err(|_| format!("spec_group_norm: flat_cols {flat_cols} exceeds u32"))?;
    let cpg_u32 = u32::try_from(channels_per_group)
        .map_err(|_| format!("spec_group_norm: channels_per_group {channels_per_group} exceeds u32"))?;
    let spatial_u32 = u32::try_from(spatial)
        .map_err(|_| format!("spec_group_norm: spatial {spatial} exceeds u32"))?;
    let num_groups_u32 = u32::try_from(num_groups)
        .map_err(|_| format!("spec_group_norm: num_groups {num_groups} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_group_norm: output bytes overflow ({total_elems} * {elem_bytes})")
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
            (4, KernelBinding::constant_u32(flat_cols_u32)),
            (5, KernelBinding::constant_f32(eps)),
            (6, KernelBinding::constant_u32(cpg_u32)),
            (7, KernelBinding::constant_u32(spatial_u32)),
            (8, KernelBinding::constant_u32(num_groups_u32)),
        ],
        param_count: 3,
        fast_math: false,
    })
}

// -------------------------------------------------------------------------
// RmsNorm
// -------------------------------------------------------------------------

/// Build a [`KernelSpec`] for fused RmsNorm.
///
/// Single Metal dispatch: `x * rsqrt(mean(x²) + eps) * weight`.
/// One threadgroup per row (flat_rows = product of dims except last).
///
/// Buffer layout:
///   0: x `[..rows, hidden_dim]` (Edge 0)
///   1: weight `[hidden_dim]` (Weight "weight")
///   2: output (Output)
///   3: hidden_dim (Constant u32)
///   4: eps (Constant f32)
///
/// Part of #3503 D3.
pub(crate) fn spec_rms_norm(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
    hidden_dim: usize,
) -> Result<KernelSpec, String> {
    if input_shape.is_empty() {
        return Err("spec_rms_norm: need rank >= 1, got 0".into());
    }

    let flat_rows: usize = if input_shape.len() == 1 {
        1
    } else {
        input_shape[..input_shape.len() - 1].iter().product()
    };
    let total_elems = flat_rows.checked_mul(hidden_dim).ok_or_else(|| {
        format!("spec_rms_norm: total overflow ({flat_rows} * {hidden_dim})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_rms_norm_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::rms_norm_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::rms_norm_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_rms_norm: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_rms_norm: flat_rows {flat_rows} exceeds u32"))?;
    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| format!("spec_rms_norm: hidden_dim {hidden_dim} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_rms_norm: output bytes overflow ({total_elems} * {elem_bytes})")
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
            (2, KernelBinding::Output),
            (3, KernelBinding::constant_u32(hidden_dim_u32)),
            (4, KernelBinding::constant_f32(eps)),
        ],
        param_count: 2,
        fast_math: false,
    })
}

// -------------------------------------------------------------------------
// Snake (standalone, not AdaIN)
// -------------------------------------------------------------------------

/// Build a [`KernelSpec`] for fused per-channel Snake activation.
///
/// Single Metal dispatch: `x + (1/alpha) * sin²(alpha * x)`.
/// Elementwise with per-channel alpha broadcast. Threadgroup count =
/// ceil(total_elems / 256).
///
/// Buffer layout:
///   0: x (Edge 0)
///   1: alpha `[C]` (Weight "alpha")
///   2: output (Output)
///   3: total_elems (Constant u32)
///   4: channel_stride (Constant u32) — product of spatial dims
///   5: channels (Constant u32)
///
/// Part of #3503 D3.
pub(crate) fn spec_snake(
    scalar_type: nn_dsl::ir::ScalarType,
    input_shape: &[usize],
    channels: usize,
) -> Result<KernelSpec, String> {
    let total_elems: usize = input_shape.iter().product();
    if total_elems == 0 {
        return Err("spec_snake: empty tensor".into());
    }

    // Channel stride = product of spatial dims (dims after channel dim).
    let channel_stride = if input_shape.len() >= 3 {
        input_shape[2..].iter().product::<usize>().max(1)
    } else {
        1
    };

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_snake_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::snake_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::snake_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_snake: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let total_elems_u32 = u32::try_from(total_elems)
        .map_err(|_| format!("spec_snake: total_elems {total_elems} exceeds u32"))?;
    let channel_stride_u32 = u32::try_from(channel_stride)
        .map_err(|_| format!("spec_snake: channel_stride {channel_stride} exceeds u32"))?;
    let channels_u32 = u32::try_from(channels)
        .map_err(|_| format!("spec_snake: channels {channels} exceeds u32"))?;
    let num_threadgroups = total_elems_u32.div_ceil(NORM_TG_SIZE);

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_snake: output bytes overflow ({total_elems} * {elem_bytes})")
    })?;

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [num_threadgroups, 1, 1],
        threadgroup: [NORM_TG_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: 0,
        output_bytes,
        bindings: vec![
            (0, KernelBinding::Edge(0)),
            (1, KernelBinding::Weight("alpha".into())),
            (2, KernelBinding::Output),
            (3, KernelBinding::constant_u32(total_elems_u32)),
            (4, KernelBinding::constant_u32(channel_stride_u32)),
            (5, KernelBinding::constant_u32(channels_u32)),
        ],
        param_count: 2,
        fast_math: false,
    })
}

// -------------------------------------------------------------------------
// FlashAttention
// -------------------------------------------------------------------------

/// Q block size for Flash Attention (threads per threadgroup).
/// Must match `BLOCK_SIZE` in `dyn_tensor_metal_flash_attn.rs`.
const FLASH_ATTN_BLOCK_SIZE: u32 = 32;

/// Build a [`KernelSpec`] for fused Flash Attention.
///
/// Single Metal dispatch: `softmax(Q @ K^T * scale) @ V`.
/// Grid: `[ceil(S_q / BLOCK_SIZE), B*H_q, 1]`. Threadgroup: `[BLOCK_SIZE, 1, 1]`.
///
/// Buffer layout:
///   0: Q `[B, H_q, S_q, D]` (Edge 0)
///   1: K `[B, H_kv, S_kv, D]` (Edge 1)
///   2: V `[B, H_kv, S_kv, D]` (Edge 2)
///   3: output (Output)
///   4: S_q (Constant u32)
///   5: S_kv (Constant u32)
///   6: D (Constant u32)
///   7: scale_bits (Constant u32) — f32 bits of scale
///   8: H_q (Constant u32)
///   9: group_size (Constant u32) — H_q / H_kv for GQA
///  10: causal (Constant u32) — 0 or 1
///
/// Part of #3503 D3.
pub(crate) fn spec_flash_attention(
    scalar_type: nn_dsl::ir::ScalarType,
    scale: f32,
    causal: bool,
    q_shape: &[usize],
    k_shape: &[usize],
    input_layout: nn_dsl::AttentionLayout,
) -> Result<KernelSpec, String> {
    if q_shape.len() != 4 || k_shape.len() != 4 {
        return Err(format!(
            "spec_flash_attention: need rank 4 for Q and K, got Q={} K={}",
            q_shape.len(),
            k_shape.len()
        ));
    }

    // Extract dimensions based on layout.
    let (batch, h_q, s_q, d, h_kv, s_kv) = match input_layout {
        nn_dsl::AttentionLayout::HeadsFirst => {
            // Q: [B, H_q, S_q, D], K: [B, H_kv, S_kv, D]
            (q_shape[0], q_shape[1], q_shape[2], q_shape[3], k_shape[1], k_shape[2])
        }
        nn_dsl::AttentionLayout::SeqFirst => {
            // Q: [B, S_q, H_q, D], K: [B, S_kv, H_kv, D]
            (q_shape[0], q_shape[2], q_shape[1], q_shape[3], k_shape[2], k_shape[1])
        }
        _ => {
            return Err(format!(
                "spec_flash_attention: unsupported layout {input_layout:?}"
            ))
        }
    };

    if h_kv == 0 || h_q % h_kv != 0 {
        return Err(format!(
            "spec_flash_attention: H_q ({h_q}) must be a multiple of H_kv ({h_kv})"
        ));
    }
    let group_size = h_q / h_kv;

    let _scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();
    let is_half = matches!(scalar_type, nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16);

    let (kernel_name, msl_source) = match input_layout {
        nn_dsl::AttentionLayout::HeadsFirst => {
            if is_half {
                ("flash_attn_f16".to_string(),
                 crate::dyn_tensor_metal::flash_attn_f16_msl_source().to_string())
            } else {
                ("flash_attn_f32".to_string(),
                 crate::dyn_tensor_metal::flash_attn_f32_msl_source().to_string())
            }
        }
        nn_dsl::AttentionLayout::SeqFirst => {
            if is_half {
                ("flash_attn_f16_seq_first".to_string(),
                 crate::dyn_tensor_metal::flash_attn_f16_seq_first_msl_source().to_string())
            } else {
                ("flash_attn_f32_seq_first".to_string(),
                 crate::dyn_tensor_metal::flash_attn_f32_seq_first_msl_source().to_string())
            }
        }
        _ => unreachable!(), // Already handled above.
    };

    let bh_q = batch.checked_mul(h_q).ok_or_else(|| {
        format!("spec_flash_attention: B*H_q overflow ({batch} * {h_q})")
    })?;
    let total_output = bh_q.checked_mul(s_q).ok_or_else(|| {
        format!("spec_flash_attention: B*H_q*S_q overflow ({bh_q} * {s_q})")
    })?.checked_mul(d).ok_or_else(|| {
        "spec_flash_attention: total output overflow".to_string()
    })?;

    let s_q_u32 = u32::try_from(s_q)
        .map_err(|_| format!("spec_flash_attention: S_q {s_q} exceeds u32"))?;
    let s_kv_u32 = u32::try_from(s_kv)
        .map_err(|_| format!("spec_flash_attention: S_kv {s_kv} exceeds u32"))?;
    let d_u32 = u32::try_from(d)
        .map_err(|_| format!("spec_flash_attention: D {d} exceeds u32"))?;
    let bh_q_u32 = u32::try_from(bh_q)
        .map_err(|_| format!("spec_flash_attention: B*H_q {bh_q} exceeds u32"))?;
    let h_q_u32 = u32::try_from(h_q)
        .map_err(|_| format!("spec_flash_attention: H_q {h_q} exceeds u32"))?;
    let group_size_u32 = u32::try_from(group_size)
        .map_err(|_| format!("spec_flash_attention: group_size {group_size} exceeds u32"))?;

    let scale_bits = scale.to_bits();
    let causal_u32: u32 = if causal { 1 } else { 0 };
    let grid_x = s_q_u32.div_ceil(FLASH_ATTN_BLOCK_SIZE);

    let output_bytes = total_output.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_flash_attention: output bytes overflow ({total_output} * {elem_bytes})")
    })?;

    Ok(KernelSpec {
        kernel_name,
        msl_source,
        grid: [grid_x, bh_q_u32, 1],
        threadgroup: [FLASH_ATTN_BLOCK_SIZE, 1, 1],
        dispatch_mode: SpecDispatchMode::Threadgroups,
        threadgroup_memory_bytes: 0,
        output_bytes,
        bindings: vec![
            (0, KernelBinding::Edge(0)),
            (1, KernelBinding::Edge(1)),
            (2, KernelBinding::Edge(2)),
            (3, KernelBinding::Output),
            (4, KernelBinding::constant_u32(s_q_u32)),
            (5, KernelBinding::constant_u32(s_kv_u32)),
            (6, KernelBinding::constant_u32(d_u32)),
            (7, KernelBinding::constant_u32(scale_bits)),
            (8, KernelBinding::constant_u32(h_q_u32)),
            (9, KernelBinding::constant_u32(group_size_u32)),
            (10, KernelBinding::constant_u32(causal_u32)),
        ],
        param_count: 3,
        fast_math: false,
    })
}

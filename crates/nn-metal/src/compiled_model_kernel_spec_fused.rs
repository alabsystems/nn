// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Fused-op `spec_*()` builder functions.
//!
//! Covers AdaIN+Snake and AdaIN+LeakyRelu. These are style-adaptive fused
//! kernels used in Kokoro's generator and decoder blocks.
//!
//! Split from `compiled_model_kernel_spec_builders.rs` per 500-line limit
//! and design doc D5.
//!
//! Part of #3503 (KernelSpec unification).

use super::{KernelBinding, KernelSpec, SpecDispatchMode};
use super::norm::NORM_TG_SIZE;

/// Build a [`KernelSpec`] for fused AdaIN+Snake.
///
/// Single Metal dispatch: InstanceNorm(x) -> affine(gamma, beta) -> Snake(alpha).
/// One threadgroup per `batch * channels` row.
///
/// Buffer layout:
///   0: x `[B, C, *spatial]` (Edge 0)
///   1: gamma `[B, C, 1]` (Edge 1)
///   2: beta `[B, C, 1]` (Edge 2)
///   3: alpha `[C]` (Weight "alpha")
///   4: output (Output)
///   5: spatial (Constant u32)
///   6: channels (Constant u32)
///   7: eps (Constant f32)
///
/// Part of #3503 D3.
pub(crate) fn spec_adain_snake(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    input_shape: &[usize],
    channels: usize,
    residual_gamma: bool,
) -> Result<KernelSpec, String> {
    if input_shape.len() < 3 {
        return Err(format!(
            "spec_adain_snake: need rank >= 3, got {}",
            input_shape.len()
        ));
    }

    let batch = input_shape[0];
    let spatial: usize = input_shape[2..].iter().product();
    let flat_rows = batch.checked_mul(channels).ok_or_else(|| {
        format!("spec_adain_snake: batch*channels overflow ({batch} * {channels})")
    })?;
    let total_elems = flat_rows.checked_mul(spatial).ok_or_else(|| {
        format!("spec_adain_snake: total overflow ({flat_rows} * {spatial})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    // The kernel name is the same for both residual_gamma variants;
    // MSL source content differs so PipelineCache distinguishes them.
    let kernel_name = format!("fused_adain_snake_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            if residual_gamma {
                crate::dyn_tensor_metal::adain_snake_msl_source()
            } else {
                // Standard gamma requires separate MSL; currently only residual_gamma
                // is used in Kokoro. Return an error for unsupported variant.
                return Err(
                    "spec_adain_snake: non-residual gamma not yet supported in spec path"
                        .into(),
                );
            }
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            if residual_gamma {
                crate::dyn_tensor_metal::adain_snake_f16_msl_source()
            } else {
                return Err(
                    "spec_adain_snake: non-residual gamma not yet supported in spec path"
                        .into(),
                );
            }
        }
        _ => {
            return Err(format!(
                "spec_adain_snake: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_adain_snake: flat_rows {flat_rows} exceeds u32"))?;
    let spatial_u32 = u32::try_from(spatial)
        .map_err(|_| format!("spec_adain_snake: spatial {spatial} exceeds u32"))?;
    let channels_u32 = u32::try_from(channels)
        .map_err(|_| format!("spec_adain_snake: channels {channels} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!("spec_adain_snake: output bytes overflow ({total_elems} * {elem_bytes})")
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
            (3, KernelBinding::Weight("alpha".into())),
            (4, KernelBinding::Output),
            (5, KernelBinding::constant_u32(spatial_u32)),
            (6, KernelBinding::constant_u32(channels_u32)),
            (7, KernelBinding::constant_f32(eps)),
        ],
        param_count: 4,
        fast_math: false,
    })
}

/// Build a [`KernelSpec`] for fused AdaIN+LeakyRelu.
///
/// Single Metal dispatch: InstanceNorm(x) -> (1+gamma)*normed+beta -> LeakyRelu(slope).
/// One threadgroup per `batch * channels` row.
///
/// Buffer layout:
///   0: x `[B, C, *spatial]` (Edge 0)
///   1: gamma `[B, C, 1]` (Edge 1)
///   2: beta `[B, C, 1]` (Edge 2)
///   3: output (Output)
///   4: spatial (Constant u32)
///   5: eps (Constant f32)
///   6: slope (Constant f32)
///
/// Part of #3503 D3.
pub(crate) fn spec_adain_leaky_relu(
    scalar_type: nn_dsl::ir::ScalarType,
    eps: f32,
    slope: f32,
    input_shape: &[usize],
) -> Result<KernelSpec, String> {
    if input_shape.len() < 3 {
        return Err(format!(
            "spec_adain_leaky_relu: need rank >= 3, got {}",
            input_shape.len()
        ));
    }

    let batch = input_shape[0];
    let channels = input_shape[1];
    let spatial: usize = input_shape[2..].iter().product();
    let flat_rows = batch.checked_mul(channels).ok_or_else(|| {
        format!("spec_adain_leaky_relu: batch*channels overflow ({batch} * {channels})")
    })?;
    let total_elems = flat_rows.checked_mul(spatial).ok_or_else(|| {
        format!("spec_adain_leaky_relu: total overflow ({flat_rows} * {spatial})")
    })?;

    let scalar_str = scalar_type.msl_str();
    let elem_bytes = scalar_type.byte_size();

    let kernel_name = format!("fused_adain_leaky_relu_{scalar_str}");
    let msl_source = match scalar_type {
        nn_dsl::ir::ScalarType::F32 => {
            crate::dyn_tensor_metal::adain_leaky_relu_msl_source()
        }
        nn_dsl::ir::ScalarType::F16 | nn_dsl::ir::ScalarType::BF16 => {
            crate::dyn_tensor_metal::adain_leaky_relu_f16_msl_source()
        }
        _ => {
            return Err(format!(
                "spec_adain_leaky_relu: unsupported scalar type {scalar_type:?}"
            ))
        }
    };

    let flat_rows_u32 = u32::try_from(flat_rows)
        .map_err(|_| format!("spec_adain_leaky_relu: flat_rows {flat_rows} exceeds u32"))?;
    let spatial_u32 = u32::try_from(spatial)
        .map_err(|_| format!("spec_adain_leaky_relu: spatial {spatial} exceeds u32"))?;

    let output_bytes = total_elems.checked_mul(elem_bytes).ok_or_else(|| {
        format!(
            "spec_adain_leaky_relu: output bytes overflow ({total_elems} * {elem_bytes})"
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
            (3, KernelBinding::Output),
            (4, KernelBinding::constant_u32(spatial_u32)),
            (5, KernelBinding::constant_f32(eps)),
            (6, KernelBinding::constant_f32(slope)),
        ],
        param_count: 3,
        fast_math: false,
    })
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MXFP4 GEMM `spec_*()` builder: Mxfp4Gemm.
//!
//! Generates a [`KernelSpec`] for the MXFP4 dequantizing matmul kernel.
//! The kernel reads packed 4-bit weights (U8 nibble-packed), per-block
//! shared exponents (U8), F32 activations, and produces F32 output.
//!
//! Part of #2242 (MXFP4 block quantization Metal kernel).

use std::mem::size_of;

use super::{KernelBinding, KernelSpec, SpecDispatchMode};

/// Build a [`KernelSpec`] for MXFP4 dequantizing matmul.
///
/// Uses the naive per-element kernel (one thread per output element).
/// Dequantizes MXFP4 weights on-the-fly via the E1M2 LUT and shared
/// block exponents.
///
/// Buffer layout:
///   0: input `[..batch, in_features]` (Edge 0, F32)
///   1: packed_weights `[out_features, in_features/2]` (Weight "packed_weights", U8)
///   2: shared_exponents `[out_features, in_features/block_size]` (Weight "shared_exponents", U8)
///   3: bias `[out_features]` (Weight "bias", F32, if has_bias)
///   3/4: output (Output, F32)
///
/// Part of #2242.
pub(crate) fn spec_mxfp4_matmul(
    in_features: usize,
    out_features: usize,
    block_size: usize,
    has_bias: bool,
    input_shape: &[usize],
) -> Result<KernelSpec, String> {
    let batch_size: usize = input_shape.iter().rev().skip(1).product();
    if batch_size == 0 || in_features == 0 || out_features == 0 {
        return Err("spec_mxfp4_matmul: zero-size dimension".into());
    }
    if in_features % block_size != 0 {
        return Err(format!(
            "spec_mxfp4_matmul: in_features ({in_features}) must be a multiple of block_size ({block_size})"
        ));
    }

    let total_output = batch_size.checked_mul(out_features).ok_or_else(|| {
        format!("spec_mxfp4_matmul: output overflow ({batch_size} * {out_features})")
    })?;

    let info = crate::compiled_model::mxfp4_gemm_msl::Mxfp4GemmInfo {
        m: batch_size,
        k: in_features,
        n: out_features,
        block_size,
        has_bias,
    };

    let msl_source = crate::compiled_model::mxfp4_gemm_msl::generate_mxfp4_gemm_msl(&info);
    let param_count = crate::compiled_model::mxfp4_gemm_msl::mxfp4_gemm_input_count(has_bias);

    let m_u32 = u32::try_from(batch_size)
        .map_err(|_| format!("spec_mxfp4_matmul: batch_size {batch_size} exceeds u32"))?;
    let n_u32 = u32::try_from(out_features)
        .map_err(|_| format!("spec_mxfp4_matmul: out_features {out_features} exceeds u32"))?;

    let tg_mem_bytes = crate::compiled_model::mxfp4_gemm_msl::mxfp4_gemm_threadgroup_bytes();

    let output_bytes = total_output
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| {
            format!("spec_mxfp4_matmul: output bytes overflow ({total_output} * 4)")
        })?;

    let mut bindings = vec![
        (0, KernelBinding::Edge(0)),
        (1, KernelBinding::Weight("packed_weights".into())),
        (2, KernelBinding::Weight("shared_exponents".into())),
    ];
    if has_bias {
        bindings.push((3, KernelBinding::Weight("bias".into())));
        bindings.push((4, KernelBinding::Output));
    } else {
        bindings.push((3, KernelBinding::Output));
    }

    // Naive per-element dispatch: one thread per output element.
    // Grid = [N, M, 1] threads, threadgroup = [16, 16, 1].
    let tg_x = 16u32;
    let tg_y = 16u32;

    Ok(KernelSpec {
        kernel_name: "mxfp4_matmul_dequant".to_string(),
        msl_source,
        grid: [n_u32.div_ceil(tg_x) * tg_x, m_u32.div_ceil(tg_y) * tg_y, 1],
        threadgroup: [tg_x, tg_y, 1],
        dispatch_mode: SpecDispatchMode::Threads,
        threadgroup_memory_bytes: tg_mem_bytes,
        output_bytes,
        bindings,
        param_count,
        fast_math: false,
    })
}

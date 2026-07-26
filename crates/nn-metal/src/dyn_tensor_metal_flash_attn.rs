// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Flash Attention 2 dispatch for Metal.
//!
//! Fuses the entire SDPA pipeline (Q@K^T*scale → softmax → @V) into a single
//! GPU dispatch using online softmax (Tri Dao, arXiv:2307.08691). Avoids
//! materializing the O(S_q × S_kv) attention matrix.
//!
//! Supports GQA (grouped-query attention) and causal masking.
//!
//! Issue: #2434

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::dispatch_plan::DispatchMode;
use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::metal_err;

use super::MetalTensorData;

/// Q block size — threads per threadgroup.
const BLOCK_SIZE: u32 = 32;

#[path = "dyn_tensor_metal_flash_attn_msl.rs"]
mod msl_kernels;
use msl_kernels::{FLASH_ATTN_F16_MSL, FLASH_ATTN_F32_MSL};

#[path = "dyn_tensor_metal_flash_attn_seq_first_msl.rs"]
mod seq_first_msl;
use seq_first_msl::{FLASH_ATTN_F16_SEQ_FIRST_MSL, FLASH_ATTN_F32_SEQ_FIRST_MSL};

/// Dimension layout for Flash Attention inputs.
enum FlashAttnLayout {
    /// Q/K/V in [B, H, S, D] — standard head-first layout.
    HeadFirst,
    /// Q/K/V in [B, S, H, D] — sequence-first layout.
    SeqFirst,
}

/// Pre-validated Flash Attention parameters.
struct FlashAttnParams {
    dtype: DType,
    is_half: bool,
    total_output: usize,
    buffer_bytes: usize,
    out_shape: Vec<usize>,
    // Pre-computed u32 values for Metal dispatch.
    s_q_u32: u32,
    s_kv_u32: u32,
    d_u32: u32,
    bh_q_u32: u32,
    h_q_u32: u32,
    group_size_u32: u32,
    scale_bits: u32,
    causal_u32: u32,
}

/// Validate Flash Attention inputs and extract pre-computed parameters.
///
/// Shared between head-first `[B, H, S, D]` and seq-first `[B, S, H, D]` layouts.
fn validate_flash_attn(
    q: &DynTensor,
    k: &DynTensor,
    v: &DynTensor,
    scale: f64,
    causal: bool,
    layout: FlashAttnLayout,
) -> Result<FlashAttnParams> {
    let label = match layout {
        FlashAttnLayout::HeadFirst => "flash_attention",
        FlashAttnLayout::SeqFirst => "flash_attention_seq_first",
    };

    // -- Dtype validation: F32, BF16, F16 ----------------------------------------
    let dtype = q.dtype();
    let is_float = matches!(dtype, DType::F32 | DType::BF16 | DType::F16);
    if !is_float {
        return Err(TensorError::dtype_mismatch(DType::F32, dtype));
    }
    if k.dtype() != dtype || v.dtype() != dtype {
        return Err(TensorError::InvalidShape(format!(
            "{label}: Q/K/V dtypes must match, got Q={:?} K={:?} V={:?}",
            dtype,
            k.dtype(),
            v.dtype()
        )));
    }
    let is_half = dtype == DType::BF16 || dtype == DType::F16;
    let bytes_per_elem: usize = if is_half { 2 } else { 4 };

    // -- Shape validation --------------------------------------------------------
    let q_shape = q.dims();
    let k_shape = k.dims();
    let v_shape = v.dims();

    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        let layout_str = match layout {
            FlashAttnLayout::HeadFirst => "[B,H,S,D]",
            FlashAttnLayout::SeqFirst => "[B,S,H,D]",
        };
        return Err(TensorError::InvalidShape(format!(
            "{label}: expected 4D {layout_str}, got Q={q_shape:?} K={k_shape:?} V={v_shape:?}"
        )));
    }

    // Layout-dependent dimension extraction — the ONLY difference.
    let (b, h_q, s_q, d, h_kv, s_kv, v_h_idx, v_s_idx) = match layout {
        FlashAttnLayout::HeadFirst => (
            q_shape[0], q_shape[1], q_shape[2], q_shape[3], k_shape[1], k_shape[2], 1usize, 2usize,
        ),
        FlashAttnLayout::SeqFirst => (
            q_shape[0], q_shape[2], q_shape[1], q_shape[3], k_shape[2], k_shape[1], 2usize, 1usize,
        ),
    };

    // Batch dims must match.
    if k_shape[0] != b || v_shape[0] != b {
        return Err(TensorError::InvalidShape(format!(
            "{label}: batch mismatch Q.B={b} K.B={} V.B={}",
            k_shape[0], v_shape[0]
        )));
    }
    // K and V head counts must match.
    if v_shape[v_h_idx] != h_kv {
        return Err(TensorError::InvalidShape(format!(
            "{label}: K/V head mismatch K.H={h_kv} V.H={}",
            v_shape[v_h_idx]
        )));
    }
    // GQA: H_q must be a multiple of H_kv.
    if h_kv == 0 || h_q % h_kv != 0 {
        return Err(TensorError::InvalidShape(format!(
            "{label}: H_q={h_q} must be a multiple of H_kv={h_kv}"
        )));
    }
    // K and V sequence lengths must match.
    if v_shape[v_s_idx] != s_kv {
        return Err(TensorError::InvalidShape(format!(
            "{label}: K/V seq mismatch K.S_kv={s_kv} V.S_kv={}",
            v_shape[v_s_idx]
        )));
    }
    // Head dim must match across Q, K, V.
    if k_shape[3] != d || v_shape[3] != d {
        return Err(TensorError::InvalidShape(format!(
            "{label}: head_dim mismatch Q.D={d} K.D={} V.D={}",
            k_shape[3], v_shape[3]
        )));
    }
    // head_dim <= 128 (kernel MAX_D constant) and > 0.
    if d == 0 || d > 128 {
        return Err(TensorError::InvalidShape(format!(
            "{label}: head_dim={d} must be in 1..=128"
        )));
    }
    // S_kv must be > 0 — with zero K/V rows the kernel loop never executes
    // and the output buffer would contain uninitialized memory.
    if s_kv == 0 {
        return Err(TensorError::InvalidShape(format!(
            "{label}: S_kv=0 (empty key/value sequence)"
        )));
    }
    // Causal requires S_q == S_kv (standard self-attention masking).
    if causal && s_q != s_kv {
        return Err(TensorError::InvalidShape(format!(
            "{label}: causal requires S_q==S_kv, got S_q={s_q} S_kv={s_kv}"
        )));
    }

    // -- Scale validation --------------------------------------------------------
    if !scale.is_finite() {
        return Err(TensorError::ValueOutOfRange {
            description: match layout {
                FlashAttnLayout::HeadFirst => "flash_attention: scale must be finite",
                FlashAttnLayout::SeqFirst => "flash_attention_seq_first: scale must be finite",
            },
        });
    }
    let scale_f32 = scale as f32;

    // -- Output dimensions -------------------------------------------------------
    let out_shape = match layout {
        FlashAttnLayout::HeadFirst => vec![b, h_q, s_q, d],
        FlashAttnLayout::SeqFirst => vec![b, s_q, h_q, d],
    };
    let total_output = b
        .checked_mul(h_q)
        .and_then(|v| v.checked_mul(s_q))
        .and_then(|v| v.checked_mul(d))
        .ok_or(TensorError::DimensionOverflow {
            dims: out_shape.clone(),
        })?;
    let buffer_bytes =
        total_output
            .checked_mul(bytes_per_elem)
            .ok_or_else(|| TensorError::DimensionOverflow {
                dims: out_shape.clone(),
            })?;

    // -- Convert dimensions to u32 for Metal dispatch ----------------------------
    let group_size = h_q / h_kv;
    let try_u32 = |val: usize, desc: &'static str| -> Result<u32> {
        u32::try_from(val).map_err(|_| TensorError::ValueOutOfRange { description: desc })
    };
    let (s_q_msg, s_kv_msg, d_msg, bh_q_msg, h_q_msg, gs_msg) = match layout {
        FlashAttnLayout::HeadFirst => (
            "flash_attention: S_q exceeds u32::MAX",
            "flash_attention: S_kv exceeds u32::MAX",
            "flash_attention: D exceeds u32::MAX",
            "flash_attention: B*H_q exceeds u32::MAX",
            "flash_attention: H_q exceeds u32::MAX",
            "flash_attention: group_size exceeds u32::MAX",
        ),
        FlashAttnLayout::SeqFirst => (
            "flash_attention_seq_first: S_q exceeds u32::MAX",
            "flash_attention_seq_first: S_kv exceeds u32::MAX",
            "flash_attention_seq_first: D exceeds u32::MAX",
            "flash_attention_seq_first: B*H_q exceeds u32::MAX",
            "flash_attention_seq_first: H_q exceeds u32::MAX",
            "flash_attention_seq_first: group_size exceeds u32::MAX",
        ),
    };

    Ok(FlashAttnParams {
        dtype,
        is_half,
        total_output,
        buffer_bytes,
        out_shape,
        s_q_u32: try_u32(s_q, s_q_msg)?,
        s_kv_u32: try_u32(s_kv, s_kv_msg)?,
        d_u32: try_u32(d, d_msg)?,
        bh_q_u32: try_u32(b * h_q, bh_q_msg)?,
        h_q_u32: try_u32(h_q, h_q_msg)?,
        group_size_u32: try_u32(group_size, gs_msg)?,
        scale_bits: scale_f32.to_bits(),
        causal_u32: if causal { 1 } else { 0 },
    })
}

/// Dispatch flash attention kernel with validated parameters.
fn dispatch_flash_attn(
    p: &FlashAttnParams,
    q_data: &MetalTensorData,
    k_data: &MetalTensorData,
    v_data: &MetalTensorData,
    msl_source: &str,
    kernel_name: &str,
) -> Result<DynTensor> {
    super::with_pipeline_cache(|cache| {
        let pipeline = KernelPipeline::from_msl(cache, msl_source, kernel_name, 3, false)
            .map_err(metal_err)?;

        let ctx = crate::metal_backend::global_metal_context().map_err(metal_err)?;
        let (out_buf, out_offset) =
            crate::arena::arena_alloc_or_create(ctx, p.buffer_bytes).map_err(metal_err)?;

        let grid_x = p.s_q_u32.div_ceil(BLOCK_SIZE);
        let grid_y = p.bh_q_u32;

        let plan = DispatchMode::Grid3D {
            grid: [grid_x, grid_y, 1],
            threads: [BLOCK_SIZE, 1, 1],
        }
        .plan()
        .map_err(metal_err)?
        .with_output_elems(p.total_output)
        .with_constants(vec![
            p.s_q_u32,
            p.s_kv_u32,
            p.d_u32,
            p.scale_bits,
            p.h_q_u32,
            p.group_size_u32,
            p.causal_u32,
        ])
        .with_use_threadgroups(true);

        pipeline
            .dispatch_buffers_with_all_offsets(
                ctx,
                &[&q_data.buffer, &k_data.buffer, &v_data.buffer],
                &[q_data.byte_offset, k_data.byte_offset, v_data.byte_offset],
                &out_buf,
                out_offset,
                &plan,
            )
            .map_err(metal_err)?;

        let storage = MetalTensorData::from_arena_alloc(out_buf, out_offset);
        DynTensor::from_gpu_storage(
            p.out_shape.clone(),
            p.dtype,
            Arc::new(storage),
            Device::metal(),
        )
    })
}

impl super::MetalDynBackend {
    /// Fused Flash Attention on GPU.
    ///
    /// Inputs:
    /// - `q`: `[B, H_q, S_q, D]`
    /// - `k`: `[B, H_kv, S_kv, D]` (H_kv ≤ H_q for GQA)
    /// - `v`: `[B, H_kv, S_kv, D]`
    /// - `scale`: typically `1/sqrt(D)`
    /// - `causal`: if true, applies causal masking (k_col > q_row → masked)
    ///
    /// Returns output `[B, H_q, S_q, D]`.
    pub(super) fn gpu_flash_attention(
        q: &DynTensor,
        k: &DynTensor,
        v: &DynTensor,
        scale: f64,
        causal: bool,
    ) -> Result<DynTensor> {
        let p = validate_flash_attn(q, k, v, scale, causal, FlashAttnLayout::HeadFirst)?;
        let q_data = q.gpu_data::<MetalTensorData>()?;
        let k_data = k.gpu_data::<MetalTensorData>()?;
        let v_data = v.gpu_data::<MetalTensorData>()?;
        let (msl_source, kernel_name) = if p.is_half {
            (FLASH_ATTN_F16_MSL, "flash_attn_f16")
        } else {
            (FLASH_ATTN_F32_MSL, "flash_attn_f32")
        };
        dispatch_flash_attn(&p, q_data, k_data, v_data, msl_source, kernel_name)
    }

    /// Fused Flash Attention — SeqFirst layout `[B, S, H, D]`.
    ///
    /// Same algorithm as `gpu_flash_attention` but expects Q/K/V in
    /// `[B, S, H, D]` layout. The MSL kernel uses stride-based addressing
    /// to avoid requiring Transpose dispatches. Part of #1815 Tier 5 D1.
    pub(super) fn gpu_flash_attention_seq_first(
        q: &DynTensor,
        k: &DynTensor,
        v: &DynTensor,
        scale: f64,
        causal: bool,
    ) -> Result<DynTensor> {
        let p = validate_flash_attn(q, k, v, scale, causal, FlashAttnLayout::SeqFirst)?;
        let q_data = q.gpu_data::<MetalTensorData>()?;
        let k_data = k.gpu_data::<MetalTensorData>()?;
        let v_data = v.gpu_data::<MetalTensorData>()?;
        let (msl_source, kernel_name) = if p.is_half {
            (FLASH_ATTN_F16_SEQ_FIRST_MSL, "flash_attn_f16_seq_first")
        } else {
            (FLASH_ATTN_F32_SEQ_FIRST_MSL, "flash_attn_f32_seq_first")
        };
        dispatch_flash_attn(&p, q_data, k_data, v_data, msl_source, kernel_name)
    }
}

/// MSL source for pre-compilation: Flash Attention f32 kernel.
pub(crate) fn flash_attn_f32_msl_source() -> &'static str {
    FLASH_ATTN_F32_MSL
}

/// MSL source for pre-compilation: Flash Attention f16 kernel.
pub(crate) fn flash_attn_f16_msl_source() -> &'static str {
    FLASH_ATTN_F16_MSL
}

/// MSL source for pre-compilation: Flash Attention f32 SeqFirst kernel.
pub(crate) fn flash_attn_f32_seq_first_msl_source() -> &'static str {
    FLASH_ATTN_F32_SEQ_FIRST_MSL
}

/// MSL source for pre-compilation: Flash Attention f16 SeqFirst kernel.
pub(crate) fn flash_attn_f16_seq_first_msl_source() -> &'static str {
    FLASH_ATTN_F16_SEQ_FIRST_MSL
}

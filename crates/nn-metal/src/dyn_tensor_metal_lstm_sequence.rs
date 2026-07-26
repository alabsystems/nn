// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused LSTM sequence kernel for [`MetalDynBackend`].
//!
//! Processes the entire `[seq_len, batch, input_size]` input in a single
//! Metal compute dispatch, eliminating per-timestep `commit_and_wait()` CPU
//! sync barriers. For Kokoro BiLSTM (5 layers, ~70 timesteps): reduces ~700
//! dispatches to ~10 (2 per layer, forward + backward directions).
//!
//! Thread grid: `[batch_size, hidden_size]`. Each thread computes one hidden
//! unit across all timesteps. Threadgroup memory shares the h vector for
//! the `w_hh @ h` dot product.
//!
//! Part of #1805: fused LSTM sequence Metal kernel.

use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result, TensorError};

use crate::kernel_dispatch::KernelPipeline;
use crate::metal_backend::{checked_dim_product, metal_err};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_lstm_msl.rs"]
mod lstm_msl;
use lstm_msl::{
    lstm_sequence_msl, lstm_sequence_precomputed_mixed_msl, lstm_sequence_precomputed_msl,
};

/// Maximum hidden_size for threadgroup memory approach.
/// 512 * 4 bytes = 2KB per batch element, well within 32KB Apple Silicon limit.
pub(crate) const MAX_THREADGROUP_HIDDEN: usize = 512;

/// Dispatch the LSTM sequence MSL kernel.
///
/// Uses raw encoder calls (like topk) because the kernel has 3 output buffers
/// (output, h_n, c_n) which the standard single-output API cannot express.
///
/// When `reverse` is true, the kernel processes input timesteps in reverse
/// order and writes output in reverse order. This eliminates external
/// `flip(dim=0)` dispatches for BiLSTM backward direction (#1815).
fn dispatch_lstm_sequence(
    input_data: &MetalTensorData,
    w_ih_data: &MetalTensorData,
    w_hh_data: &MetalTensorData,
    bias_data: Option<&MetalTensorData>,
    h0_data: &MetalTensorData,
    c0_data: &MetalTensorData,
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
    reverse: bool,
) -> Result<(DynTensor, DynTensor, DynTensor)> {
    let ctx = super::MetalDynBackend::ctx()?;
    let out_numel = checked_dim_product(&[seq_len, batch_size, hidden_size])?;
    let state_numel = checked_dim_product(&[batch_size, hidden_size])?;

    super::with_pipeline_cache(|cache| {
        let msl = lstm_sequence_msl(hidden_size);
        let pipeline = KernelPipeline::from_msl(cache, &msl, "lstm_forward_sequence", 1, false)
            .map_err(metal_err)?;

        // Allocate output buffers WITHOUT arena (#2659).
        //
        // LSTM sequence outputs are primary outputs consumed by downstream ops
        // (e.g., BiLSTM cat) that may span across flush boundaries. The default
        // arena resets on flush(), so arena-backed outputs become stale when a
        // subsequent operation (e.g., backward LSTM weight validation) triggers
        // flush(). Using without_arena() ensures standalone Metal buffers that
        // survive arena resets.
        let out_bytes = out_numel.checked_mul(size_of::<f32>()).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: vec![seq_len, batch_size, hidden_size],
            }
        })?;
        let state_bytes = state_numel.checked_mul(size_of::<f32>()).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: vec![batch_size, hidden_size],
            }
        })?;
        let (out_buf, out_offset) =
            crate::arena::without_arena(|| crate::arena::arena_alloc_or_create(ctx, out_bytes))
                .map_err(metal_err)?;
        let out_arena_gen = crate::arena::last_alloc_generation();
        let (h_n_buf, h_n_offset) =
            crate::arena::without_arena(|| crate::arena::arena_alloc_or_create(ctx, state_bytes))
                .map_err(metal_err)?;
        let h_n_arena_gen = crate::arena::last_alloc_generation();
        let (c_n_buf, c_n_offset) =
            crate::arena::without_arena(|| crate::arena::arena_alloc_or_create(ctx, state_bytes))
                .map_err(metal_err)?;
        let c_n_arena_gen = crate::arena::last_alloc_generation();

        // Create a zero-length bias buffer when no bias is provided.
        // The kernel checks `has_bias` before reading, so this is safe.
        let empty_bias;
        let bias_buf = match bias_data {
            Some(b) => &b.buffer,
            None => {
                empty_bias = ctx.create_buffer_zeroed(4).map_err(metal_err)?;
                &empty_bias
            }
        };

        let seq_len_u32 = crate::to_u32(seq_len, "lstm seq_len")?;
        let batch_size_u32 = crate::to_u32(batch_size, "lstm batch_size")?;
        let input_size_u32 = crate::to_u32(input_size, "lstm input_size")?;
        let hidden_size_u32 = crate::to_u32(hidden_size, "lstm hidden_size")?;
        let has_bias_u32: u32 = if bias_data.is_some() { 1 } else { 0 };
        let reverse_u32: u32 = u32::from(reverse);
        let bias_byte_offset = bias_data.as_ref().map_or(0, |b| b.byte_offset);

        // Encode buffer bindings and dispatch into a compute encoder.
        // Works with both BatchEncoder (GpuScope) and ComputeDispatch (standalone).
        macro_rules! encode_lstm {
            ($enc:expr) => {{
                $enc.set_buffer_with_offset(0, &input_data.buffer, input_data.byte_offset);
                $enc.set_buffer_with_offset(1, &w_ih_data.buffer, w_ih_data.byte_offset);
                $enc.set_buffer_with_offset(2, &w_hh_data.buffer, w_hh_data.byte_offset);
                $enc.set_buffer_with_offset(3, bias_buf, bias_byte_offset);
                $enc.set_buffer_with_offset(4, &h0_data.buffer, h0_data.byte_offset);
                $enc.set_buffer_with_offset(5, &c0_data.buffer, c0_data.byte_offset);
                $enc.set_buffer_with_offset(6, &out_buf, out_offset);
                $enc.set_buffer_with_offset(7, &h_n_buf, h_n_offset);
                $enc.set_buffer_with_offset(8, &c_n_buf, c_n_offset);
                $enc.set_bytes(9, &seq_len_u32);
                $enc.set_bytes(10, &batch_size_u32);
                $enc.set_bytes(11, &input_size_u32);
                $enc.set_bytes(12, &hidden_size_u32);
                $enc.set_bytes(13, &has_bias_u32);
                $enc.set_bytes(14, &reverse_u32);
                // Thread grid: one threadgroup per batch element, hidden_size threads per group.
                $enc.encode_threadgroups(
                    pipeline.pipeline(),
                    [batch_size_u32, 1, 1],
                    [hidden_size_u32, 1, 1],
                )
            }};
        }

        // Lazy batch (#2009): encode into the thread-local lazy batch.
        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> std::result::Result<(), crate::error::MetalError> {
                let enc = batch.new_encoder()?;
                encode_lstm!(enc)?;
                enc.end_encoding();
                Ok(())
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Wrap output buffers as DynTensors with arena generation stamps (#2328).
        let out_storage = match out_arena_gen {
            Some(g) => MetalTensorData::view_arena(out_buf.alias(), out_offset, g),
            None if out_offset > 0 => MetalTensorData::view(out_buf.alias(), out_offset),
            None => MetalTensorData::new(out_buf),
        };
        let h_n_storage = match h_n_arena_gen {
            Some(g) => MetalTensorData::view_arena(h_n_buf.alias(), h_n_offset, g),
            None if h_n_offset > 0 => MetalTensorData::view(h_n_buf.alias(), h_n_offset),
            None => MetalTensorData::new(h_n_buf),
        };
        let c_n_storage = match c_n_arena_gen {
            Some(g) => MetalTensorData::view_arena(c_n_buf.alias(), c_n_offset, g),
            None if c_n_offset > 0 => MetalTensorData::view(c_n_buf.alias(), c_n_offset),
            None => MetalTensorData::new(c_n_buf),
        };

        let output = DynTensor::from_gpu_storage(
            vec![seq_len, batch_size, hidden_size],
            DType::F32,
            Arc::new(out_storage),
            Device::metal(),
        )?;
        let h_n = DynTensor::from_gpu_storage(
            vec![batch_size, hidden_size],
            DType::F32,
            Arc::new(h_n_storage),
            Device::metal(),
        )?;
        let c_n = DynTensor::from_gpu_storage(
            vec![batch_size, hidden_size],
            DType::F32,
            Arc::new(c_n_storage),
            Device::metal(),
        )?;

        Ok((output, h_n, c_n))
    })
}

/// Dispatch the precomputed-input LSTM sequence MSL kernel.
///
/// Takes pre-projected `input_proj` [seq_len, batch, 4*hidden_size] instead of
/// raw input + w_ih. The input projection `X @ W_ih.T + bias` is computed
/// externally via simdgroup matmul (parallel across all timesteps), so this
/// kernel only does the sequential `w_hh @ h` recurrence.
///
/// Part of #2981 (LSTM input GEMM pre-computation), restored in #3491.
pub(crate) fn dispatch_lstm_precomputed(
    input_proj_data: &MetalTensorData,
    w_hh_data: &MetalTensorData,
    h0_data: &MetalTensorData,
    c0_data: &MetalTensorData,
    seq_len: usize,
    batch_size: usize,
    hidden_size: usize,
    reverse: bool,
    mixed: bool,
) -> Result<(DynTensor, DynTensor, DynTensor)> {
    let ctx = super::MetalDynBackend::ctx()?;
    let out_numel = checked_dim_product(&[seq_len, batch_size, hidden_size])?;
    let state_numel = checked_dim_product(&[batch_size, hidden_size])?;

    super::with_pipeline_cache(|cache| {
        let (msl, kernel_name) = if mixed {
            (
                lstm_sequence_precomputed_mixed_msl(hidden_size),
                "lstm_forward_sequence_precomputed_mixed",
            )
        } else {
            (
                lstm_sequence_precomputed_msl(hidden_size),
                "lstm_forward_sequence_precomputed",
            )
        };
        let pipeline =
            KernelPipeline::from_msl(cache, &msl, kernel_name, 1, false).map_err(metal_err)?;

        // Allocate output buffers WITHOUT arena (#2659) — same rationale as fused path.
        let out_bytes = out_numel.checked_mul(size_of::<f32>()).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: vec![seq_len, batch_size, hidden_size],
            }
        })?;
        let state_bytes = state_numel.checked_mul(size_of::<f32>()).ok_or_else(|| {
            TensorError::DimensionOverflow {
                dims: vec![batch_size, hidden_size],
            }
        })?;
        let (out_buf, out_offset) =
            crate::arena::without_arena(|| crate::arena::arena_alloc_or_create(ctx, out_bytes))
                .map_err(metal_err)?;
        let out_arena_gen = crate::arena::last_alloc_generation();
        let (h_n_buf, h_n_offset) =
            crate::arena::without_arena(|| crate::arena::arena_alloc_or_create(ctx, state_bytes))
                .map_err(metal_err)?;
        let h_n_arena_gen = crate::arena::last_alloc_generation();
        let (c_n_buf, c_n_offset) =
            crate::arena::without_arena(|| crate::arena::arena_alloc_or_create(ctx, state_bytes))
                .map_err(metal_err)?;
        let c_n_arena_gen = crate::arena::last_alloc_generation();

        let seq_len_u32 = crate::to_u32(seq_len, "lstm_precomputed seq_len")?;
        let batch_size_u32 = crate::to_u32(batch_size, "lstm_precomputed batch_size")?;
        let hidden_size_u32 = crate::to_u32(hidden_size, "lstm_precomputed hidden_size")?;
        let reverse_u32: u32 = u32::from(reverse);

        // Encode buffer bindings for the precomputed kernel.
        macro_rules! encode_lstm_precomputed {
            ($enc:expr) => {{
                $enc.set_buffer_with_offset(
                    0,
                    &input_proj_data.buffer,
                    input_proj_data.byte_offset,
                );
                $enc.set_buffer_with_offset(1, &w_hh_data.buffer, w_hh_data.byte_offset);
                $enc.set_buffer_with_offset(2, &h0_data.buffer, h0_data.byte_offset);
                $enc.set_buffer_with_offset(3, &c0_data.buffer, c0_data.byte_offset);
                $enc.set_buffer_with_offset(4, &out_buf, out_offset);
                $enc.set_buffer_with_offset(5, &h_n_buf, h_n_offset);
                $enc.set_buffer_with_offset(6, &c_n_buf, c_n_offset);
                $enc.set_bytes(7, &seq_len_u32);
                $enc.set_bytes(8, &batch_size_u32);
                $enc.set_bytes(9, &hidden_size_u32);
                $enc.set_bytes(10, &reverse_u32);
                $enc.encode_threadgroups(
                    pipeline.pipeline(),
                    [batch_size_u32, 1, 1],
                    [hidden_size_u32, 1, 1],
                )
            }};
        }

        // Lazy batch: encode into the thread-local lazy batch.
        crate::gpu_scope::get_or_create_batch()?;
        let scope_result = crate::gpu_scope::encode_into_lazy_batch(
            |batch| -> std::result::Result<(), crate::error::MetalError> {
                let enc = batch.new_encoder()?;
                encode_lstm_precomputed!(enc)?;
                enc.end_encoding();
                Ok(())
            },
        );
        match scope_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(metal_err(e)),
            Err(e) => return Err(e),
        }

        // Wrap output buffers as DynTensors with arena generation stamps.
        let out_storage = match out_arena_gen {
            Some(g) => MetalTensorData::view_arena(out_buf.alias(), out_offset, g),
            None if out_offset > 0 => MetalTensorData::view(out_buf.alias(), out_offset),
            None => MetalTensorData::new(out_buf),
        };
        let h_n_storage = match h_n_arena_gen {
            Some(g) => MetalTensorData::view_arena(h_n_buf.alias(), h_n_offset, g),
            None if h_n_offset > 0 => MetalTensorData::view(h_n_buf.alias(), h_n_offset),
            None => MetalTensorData::new(h_n_buf),
        };
        let c_n_storage = match c_n_arena_gen {
            Some(g) => MetalTensorData::view_arena(c_n_buf.alias(), c_n_offset, g),
            None if c_n_offset > 0 => MetalTensorData::view(c_n_buf.alias(), c_n_offset),
            None => MetalTensorData::new(c_n_buf),
        };

        let output = DynTensor::from_gpu_storage(
            vec![seq_len, batch_size, hidden_size],
            DType::F32,
            Arc::new(out_storage),
            Device::metal(),
        )?;
        let h_n = DynTensor::from_gpu_storage(
            vec![batch_size, hidden_size],
            DType::F32,
            Arc::new(h_n_storage),
            Device::metal(),
        )?;
        let c_n = DynTensor::from_gpu_storage(
            vec![batch_size, hidden_size],
            DType::F32,
            Arc::new(c_n_storage),
            Device::metal(),
        )?;

        Ok((output, h_n, c_n))
    })
}

impl super::MetalDynBackend {
    /// GPU-native LSTM sequence: processes full `[seq_len, batch, input_size]`
    /// in a single Metal dispatch.
    ///
    /// Returns `None` for hidden_size > 512 (threadgroup memory limit)
    /// or non-f32 tensors, falling back to the per-timestep loop.
    pub(super) fn gpu_lstm_sequence(
        input: &DynTensor,
        w_ih: &DynTensor,
        w_hh: &DynTensor,
        bias: Option<&DynTensor>,
        h0: &DynTensor,
        c0: &DynTensor,
        hidden_size: usize,
        skip_weight_validation: bool,
    ) -> Option<Result<(DynTensor, DynTensor, DynTensor)>> {
        if hidden_size == 0 {
            return crate::gpu_fallback(
                "lstm_sequence",
                "hidden_size=0 causes zero-length MSL threadgroup array (UB)",
            );
        }
        if hidden_size > MAX_THREADGROUP_HIDDEN {
            return crate::gpu_fallback(
                "lstm_sequence",
                "hidden_size > 512 exceeds threadgroup memory limit",
            );
        }
        if Self::validate_f32_buffer(input, "gpu_lstm_sequence").is_err() {
            return crate::gpu_fallback("lstm_sequence", "non-f32 input");
        }
        Some(Self::gpu_lstm_sequence_impl(
            input,
            w_ih,
            w_hh,
            bias,
            h0,
            c0,
            hidden_size,
            skip_weight_validation,
            false, // forward
        ))
    }

    /// Reverse-direction LSTM sequence for BiLSTM backward pass (#1815).
    ///
    /// Same validation as [`gpu_lstm_sequence`] but the kernel reads input
    /// timesteps from `seq_len-1` down to 0 and writes output in the same
    /// reversed order. Eliminates 2 external `flip(dim=0)` dispatches per
    /// BiLSTM layer.
    pub(super) fn gpu_lstm_sequence_reverse(
        input: &DynTensor,
        w_ih: &DynTensor,
        w_hh: &DynTensor,
        bias: Option<&DynTensor>,
        h0: &DynTensor,
        c0: &DynTensor,
        hidden_size: usize,
        skip_weight_validation: bool,
    ) -> Option<Result<(DynTensor, DynTensor, DynTensor)>> {
        if hidden_size == 0 {
            return crate::gpu_fallback(
                "lstm_sequence_reverse",
                "hidden_size=0 causes zero-length MSL threadgroup array (UB)",
            );
        }
        if hidden_size > MAX_THREADGROUP_HIDDEN {
            return crate::gpu_fallback(
                "lstm_sequence_reverse",
                "hidden_size > 512 exceeds threadgroup memory limit",
            );
        }
        if Self::validate_f32_buffer(input, "gpu_lstm_sequence_reverse").is_err() {
            return crate::gpu_fallback("lstm_sequence_reverse", "non-f32 input");
        }
        Some(Self::gpu_lstm_sequence_impl(
            input,
            w_ih,
            w_hh,
            bias,
            h0,
            c0,
            hidden_size,
            skip_weight_validation,
            true, // reverse
        ))
    }

    fn gpu_lstm_sequence_impl(
        input: &DynTensor,
        w_ih: &DynTensor,
        w_hh: &DynTensor,
        bias: Option<&DynTensor>,
        h0: &DynTensor,
        c0: &DynTensor,
        hidden_size: usize,
        skip_weight_validation: bool,
        reverse: bool,
    ) -> Result<(DynTensor, DynTensor, DynTensor)> {
        // Validate input shape: [seq_len, batch, input_size].
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }
        let seq_len = dims[0];
        let batch_size = dims[1];
        let input_size = dims[2];

        // Validate all tensors share the same float dtype.
        // MSL kernel hardcodes `float*` buffer types — mixed dtypes (e.g., f32 input
        // with bf16 weights) would read garbage data silently. Matches gpu_lstm_cell pattern.
        Self::validate_same_float_dtype(input, w_ih, "gpu_lstm_sequence")?;
        Self::validate_same_float_dtype(input, w_hh, "gpu_lstm_sequence")?;
        Self::validate_same_float_dtype(input, h0, "gpu_lstm_sequence")?;
        Self::validate_same_float_dtype(input, c0, "gpu_lstm_sequence")?;
        if let Some(b) = bias {
            Self::validate_same_float_dtype(input, b, "gpu_lstm_sequence")?;
        }

        // Validate weight shapes.
        let (wih_rows, wih_cols) = w_ih.dims2().map_err(|_| TensorError::RankMismatch {
            expected: 2,
            actual: w_ih.dims().len(),
        })?;
        if wih_rows != 4 * hidden_size || wih_cols != input_size {
            return Err(TensorError::shape_mismatch(
                vec![4 * hidden_size, input_size],
                w_ih.dims().to_vec(),
            ));
        }
        let (whh_rows, whh_cols) = w_hh.dims2().map_err(|_| TensorError::RankMismatch {
            expected: 2,
            actual: w_hh.dims().len(),
        })?;
        if whh_rows != 4 * hidden_size || whh_cols != hidden_size {
            return Err(TensorError::shape_mismatch(
                vec![4 * hidden_size, hidden_size],
                w_hh.dims().to_vec(),
            ));
        }

        // Validate weight/state finiteness before kernel launch.
        // GPU kernel applies sigmoid(Inf)=1.0 and tanh(Inf)=1.0 which silently
        // absorbs Inf values. Reject non-finite weights to match CPU error behavior.
        //
        // Skipped in compiled model paths (#2795) where weights are pre-uploaded
        // GPU buffers that never change between forward passes. Each
        // any_non_finite() call forces a GPU flush to read buffer contents,
        // adding 12 unnecessary sync points per Kokoro forward pass.
        if !skip_weight_validation {
            if w_ih.any_non_finite()? {
                return Err(TensorError::NonFiniteData {
                    name: "gpu_lstm_sequence: w_ih".into(),
                    count: 1, // at least 1; exact count requires CPU readback
                });
            }
            if w_hh.any_non_finite()? {
                return Err(TensorError::NonFiniteData {
                    name: "gpu_lstm_sequence: w_hh".into(),
                    count: 1,
                });
            }
            if let Some(b) = bias {
                if b.any_non_finite()? {
                    return Err(TensorError::NonFiniteData {
                        name: "gpu_lstm_sequence: bias".into(),
                        count: 1,
                    });
                }
            }
        }

        // Validate state shapes.
        if h0.dims() != [batch_size, hidden_size] {
            return Err(TensorError::shape_mismatch(
                vec![batch_size, hidden_size],
                h0.dims().to_vec(),
            ));
        }
        if c0.dims() != [batch_size, hidden_size] {
            return Err(TensorError::shape_mismatch(
                vec![batch_size, hidden_size],
                c0.dims().to_vec(),
            ));
        }

        // Validate h0/c0 finiteness (#R1-1151 F1, defense-in-depth).
        // GPU kernel applies sigmoid(Inf)=1.0 and tanh(Inf)=1.0 which silently
        // absorbs non-finite state values, producing finite-looking but incorrect output.
        //
        // Also skipped for compiled model paths (#2795) — h0/c0 are pre-uploaded
        // zero-initialized buffers that are always finite.
        if !skip_weight_validation {
            if h0.any_non_finite()? {
                return Err(TensorError::NonFiniteData {
                    name: "gpu_lstm_sequence: h0".into(),
                    count: 1,
                });
            }
            if c0.any_non_finite()? {
                return Err(TensorError::NonFiniteData {
                    name: "gpu_lstm_sequence: c0".into(),
                    count: 1,
                });
            }
        }

        // Extract GPU buffers.
        let input_data = input.gpu_data::<MetalTensorData>()?;
        let w_ih_data = w_ih.gpu_data::<MetalTensorData>()?;
        let w_hh_data = w_hh.gpu_data::<MetalTensorData>()?;
        let bias_data = bias.map(|b| b.gpu_data::<MetalTensorData>()).transpose()?;
        let h0_data = h0.gpu_data::<MetalTensorData>()?;
        let c0_data = c0.gpu_data::<MetalTensorData>()?;

        dispatch_lstm_sequence(
            input_data,
            w_ih_data,
            w_hh_data,
            bias_data,
            h0_data,
            c0_data,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            reverse,
        )
    }
}

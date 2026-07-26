// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! BiLSTM-Cat execution for `CompiledModel`.
//!
//! Implements `NativeOpKind::BiLstmCat`: runs forward and reverse LSTMs
//! using merged weights (prefixed `fwd_`/`rev_` from the peephole pass),
//! then concatenates outputs along the last dimension.
//!
//! Part of #4252.

use std::collections::HashMap;
use std::sync::Arc;

use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device, Result};

use crate::buffer::MetalBuffer;
use crate::dyn_tensor_metal::MetalTensorData;
use crate::gpu_slice::GpuSlice;

use super::super::helpers::{dyn_to_slice, native_dispatch_err, slice_to_dyn, weight_to_dyn};
use super::super::CompiledModel;

/// Execute a `NativeOpKind::BiLstmCat` step.
///
/// Runs forward and reverse LSTMs internally using the merged weight map
/// (keys prefixed with `fwd_` and `rev_` from the peephole pass), then
/// concatenates their outputs along the last dimension.
///
/// The BiLstmCat step owns the merged weights from both original LSTM steps.
/// The `fwd_lstm_step` and `rev_lstm_step` fields are informational only
/// (used for diagnostics); all weight access goes through the BiLstmCat's
/// own `step_idx`.
///
/// Part of #4252.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_native_bilstm_cat(
    model: &CompiledModel,
    step_idx: usize,
    buffers: &[Option<GpuSlice>],
    hidden_size: usize,
    input_shape: &[usize],
    h_shape: &[usize],
    _fwd_lstm_step: usize,
    _rev_lstm_step: usize,
) -> Result<GpuSlice> {
    let input_slice = model.resolve_input_slice(step_idx, 0, buffers)?;
    let dtype = model.step_dtype(step_idx);
    let step_weights = &model.def.weight_buffers[step_idx];

    // Run forward LSTM with fwd_-prefixed weights.
    let fwd_output = run_bilstm_direction(
        step_weights,
        &input_slice,
        dtype,
        step_idx,
        hidden_size,
        input_shape,
        h_shape,
        "fwd_",
        false,
    )?;

    // Run reverse LSTM with rev_-prefixed weights.
    let rev_output = run_bilstm_direction(
        step_weights,
        &input_slice,
        dtype,
        step_idx,
        hidden_size,
        input_shape,
        h_shape,
        "rev_",
        true,
    )?;

    // Concatenate along last dimension: [S, B, H] + [S, B, H] → [S, B, 2H].
    // LSTM output shape is [S, B, hidden_size].
    let out_shape: Vec<usize> = {
        let mut s = input_shape.to_vec();
        if let Some(last) = s.last_mut() {
            *last = hidden_size;
        }
        s
    };
    let fwd_tensor = slice_to_dyn(&fwd_output, &out_shape, dtype)?;
    let rev_tensor = slice_to_dyn(&rev_output, &out_shape, dtype)?;

    let cat_dim = out_shape.len().saturating_sub(1);
    let cat_output = DynTensor::cat(&[&fwd_tensor, &rev_tensor], cat_dim)?;

    dyn_to_slice(&cat_output, step_idx, "NativeOp BiLstmCat")
}

/// Run one direction of a BiLSTM using prefixed weight keys.
///
/// Supports both the fused path (single kernel) and the precomputed GEMM path
/// (if `{prefix}weight_ih_t` is present and alignment conditions are met).
#[allow(clippy::too_many_arguments)]
fn run_bilstm_direction(
    step_weights: &HashMap<String, MetalBuffer>,
    input_slice: &GpuSlice,
    dtype: DType,
    step_idx: usize,
    hidden_size: usize,
    input_shape: &[usize],
    h_shape: &[usize],
    prefix: &str,
    reverse: bool,
) -> Result<GpuSlice> {
    let seq_len = input_shape[0];
    let batch_size = input_shape[1];
    let input_size = input_shape[2];
    let n = 4 * hidden_size;
    let dir_label = if reverse { "rev" } else { "fwd" };
    let op_name = format!("NativeOp BiLstmCat({dir_label})");

    // Check for precomputed GEMM path.
    let weight_ih_t_key = format!("{prefix}weight_ih_t");
    let has_weight_ih_t = step_weights.contains_key(&weight_ih_t_key);

    if has_weight_ih_t && input_size.is_multiple_of(8) && n.is_multiple_of(8) {
        return execute_bilstm_precomputed(
            step_weights,
            input_slice,
            dtype,
            step_idx,
            seq_len,
            batch_size,
            input_size,
            hidden_size,
            prefix,
            reverse,
        );
    }

    // Fused path: single-dispatch kernel per direction.
    let input_tensor = slice_to_dyn(input_slice, input_shape, dtype)?;

    let w_ih_key = format!("{prefix}weight_ih");
    let w_hh_key = format!("{prefix}weight_hh");
    let h0_key = format!("{prefix}h0");
    let c0_key = format!("{prefix}c0");

    let w_ih = weight_to_dyn(
        step_weights, &w_ih_key, &[n, input_size], dtype, step_idx, &op_name,
    )?;
    let w_hh = weight_to_dyn(
        step_weights, &w_hh_key, &[n, hidden_size], dtype, step_idx, &op_name,
    )?;
    let h0 = weight_to_dyn(
        step_weights, &h0_key, h_shape, dtype, step_idx, &op_name,
    )?;
    let c0 = weight_to_dyn(
        step_weights, &c0_key, h_shape, dtype, step_idx, &op_name,
    )?;
    let bias = load_combined_bias_prefixed(
        step_weights, hidden_size, dtype, step_idx, prefix,
    )?;

    let dispatch_fn = if reverse {
        crate::dyn_tensor_metal::native_lstm_sequence_reverse
    } else {
        crate::dyn_tensor_metal::native_lstm_sequence
    };
    let (output, _h_n, _c_n) = dispatch_fn(
        &input_tensor, &w_ih, &w_hh, bias.as_ref(), &h0, &c0, hidden_size,
        true,
    )
    .ok_or_else(|| {
        native_dispatch_err(
            step_idx,
            format!(
                "{op_name}: gpu_lstm_sequence{} returned None \
                 (hidden_size={hidden_size}, max=512)",
                if reverse { "_reverse" } else { "" }
            ),
        )
    })??;

    dyn_to_slice(&output, step_idx, &op_name)
}

/// Precomputed GEMM path for one direction of a BiLSTM.
///
/// Mirrors `execute_precomputed_lstm` but reads weight keys with `prefix`.
#[allow(clippy::too_many_arguments)]
fn execute_bilstm_precomputed(
    step_weights: &HashMap<String, MetalBuffer>,
    input_slice: &GpuSlice,
    dtype: DType,
    step_idx: usize,
    seq_len: usize,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
    prefix: &str,
    reverse: bool,
) -> Result<GpuSlice> {
    let m = seq_len * batch_size;
    let n = 4 * hidden_size;
    let dir_label = if reverse { "rev" } else { "fwd" };
    let op_name = format!("NativeOp BiLstmCat({dir_label}) precomputed");

    // Phase 1: input_proj = input_2d @ weight_ih_t
    let input_data = MetalTensorData::view(input_slice.buffer().alias(), input_slice.byte_offset());

    let weight_ih_t_key = format!("{prefix}weight_ih_t");
    let weight_ih_t_buf = step_weights.get(&weight_ih_t_key).ok_or_else(|| {
        native_dispatch_err(step_idx, format!("{op_name}: missing weight '{weight_ih_t_key}'"))
    })?;
    let weight_ih_t_data = MetalTensorData::new(weight_ih_t_buf.alias());

    let proj_data = crate::dyn_tensor_metal::encode_simdgroup_matmul_into_batch(
        &input_data,
        &weight_ih_t_data,
        m,
        input_size,
        n,
    )?;

    // Add bias if present.
    let bias = load_combined_bias_prefixed(
        step_weights, hidden_size, dtype, step_idx, prefix,
    )?;
    let proj_tensor = DynTensor::from_gpu_storage(
        vec![m, n],
        dtype,
        Arc::new(proj_data),
        Device::metal(),
    )?;
    let proj_with_bias = match bias {
        Some(ref b) => proj_tensor.add(b)?,
        None => proj_tensor,
    };

    let proj_3d = proj_with_bias.reshape([seq_len, batch_size, n])?;
    let proj_3d_data = proj_3d.gpu_data::<MetalTensorData>().map_err(|_| {
        native_dispatch_err(step_idx, format!("{op_name}: proj not GPU tensor"))
    })?;

    // Phase 2: precomputed LSTM recurrence.
    let w_hh_key = format!("{prefix}weight_hh");
    let w_hh_buf = step_weights.get(&w_hh_key).ok_or_else(|| {
        native_dispatch_err(step_idx, format!("{op_name}: missing '{w_hh_key}'"))
    })?;
    let w_hh_data = MetalTensorData::new(w_hh_buf.alias());

    let h0_key = format!("{prefix}h0");
    let h0_buf = step_weights.get(&h0_key).ok_or_else(|| {
        native_dispatch_err(step_idx, format!("{op_name}: missing '{h0_key}'"))
    })?;
    let h0_data = MetalTensorData::new(h0_buf.alias());

    let c0_key = format!("{prefix}c0");
    let c0_buf = step_weights.get(&c0_key).ok_or_else(|| {
        native_dispatch_err(step_idx, format!("{op_name}: missing '{c0_key}'"))
    })?;
    let c0_data = MetalTensorData::new(c0_buf.alias());

    let (output, _h_n, _c_n) = crate::dyn_tensor_metal::dispatch_lstm_precomputed(
        proj_3d_data,
        &w_hh_data,
        &h0_data,
        &c0_data,
        seq_len,
        batch_size,
        hidden_size,
        reverse,
        false,
    )?;

    dyn_to_slice(&output, step_idx, &op_name)
}

/// Combine `{prefix}bias_ih + {prefix}bias_hh` or use single `{prefix}bias`.
fn load_combined_bias_prefixed(
    step_weights: &HashMap<String, MetalBuffer>,
    hidden_size: usize,
    dtype: DType,
    step_idx: usize,
    prefix: &str,
) -> Result<Option<DynTensor>> {
    let bih_key = format!("{prefix}bias_ih");
    let bhh_key = format!("{prefix}bias_hh");
    let single_key = format!("{prefix}bias");
    let op_name = format!("NativeOp BiLstmCat({prefix}LSTM)");
    let has_bih = step_weights.contains_key(&bih_key);
    let has_bhh = step_weights.contains_key(&bhh_key);
    let has_single = step_weights.contains_key(&single_key);
    if has_bih && has_bhh {
        let bih = weight_to_dyn(
            step_weights, &bih_key, &[4 * hidden_size], dtype, step_idx, &op_name,
        )?;
        let bhh = weight_to_dyn(
            step_weights, &bhh_key, &[4 * hidden_size], dtype, step_idx, &op_name,
        )?;
        Ok(Some(bih.add(&bhh)?))
    } else if has_single {
        Ok(Some(weight_to_dyn(
            step_weights, &single_key, &[4 * hidden_size], dtype, step_idx, &op_name,
        )?))
    } else {
        Ok(None)
    }
}

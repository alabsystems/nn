// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Axis-switch dispatch helpers for the Demucs spectral decoder.
//!
//! The spectral branch operates on 4D tensors `[C, F, T]` but nn's 1D ops
//! (DConv, ConvTranspose1d) work on `[C, N]`. These helpers extract slices
//! along one spatial axis, dispatch the 1D op, and reassemble the output.
//!
//! Part of #779 Phase B.

use std::borrow::Cow;
use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::ScalarType;

use crate::tensor_dispatch::execute_tensor_dispatch_batched;
use crate::PipelineCache;

use super::DemucsSpectralDecoderError;

/// Center-trim a flattened `[C, F_current, T_current]` tensor along F and T.
///
/// Returns `[C, target_f, target_t]`, taking the center region from each
/// channel's spatial grid.
pub(super) fn center_trim_2d(
    data: &[f32],
    channels: usize,
    current_f: usize,
    current_t: usize,
    target_f: usize,
    target_t: usize,
) -> Result<Vec<f32>, DemucsSpectralDecoderError> {
    if current_f == target_f && current_t == target_t {
        return Ok(data.to_vec());
    }
    let delta_f =
        current_f
            .checked_sub(target_f)
            .ok_or_else(|| DemucsSpectralDecoderError::DimMismatch {
                stage: "center_trim_2d".into(),
                expected: current_f,
                actual: target_f,
            })?;
    let start_f = delta_f / 2;
    let delta_t =
        current_t
            .checked_sub(target_t)
            .ok_or_else(|| DemucsSpectralDecoderError::DimMismatch {
                stage: "center_trim_2d".into(),
                expected: current_t,
                actual: target_t,
            })?;
    let start_t = delta_t / 2;

    let cap = channels
        .checked_mul(target_f)
        .and_then(|v| v.checked_mul(target_t))
        .ok_or_else(|| DemucsSpectralDecoderError::DimMismatch {
            stage: "center_trim_2d allocation overflow".into(),
            expected: 0,
            actual: channels,
        })?;
    let mut out = Vec::with_capacity(cap);
    for c in 0..channels {
        let ch_offset = c * current_f * current_t;
        for f in 0..target_f {
            let row_offset = ch_offset + (start_f + f) * current_t + start_t;
            out.extend_from_slice(&data[row_offset..row_offset + target_t]);
        }
    }
    Ok(out)
}

/// Dispatch DConv independently for each frequency bin.
///
/// Input: flattened `[C, F, T]`. For each of F frequency bins, extracts
/// `[C, T]` and dispatches with the DConv def. Output: `[C, F, T]`.
pub(super) fn dispatch_per_freq_bin(
    cache: &PipelineCache,
    dconv_def: &TensorKernelDef,
    data: &[f32],
    dconv_weights: &HashMap<String, Vec<f32>>,
    channels: usize,
    freq_bins: usize,
    time_len: usize,
) -> Result<Vec<f32>, DemucsSpectralDecoderError> {
    let ct =
        channels
            .checked_mul(time_len)
            .ok_or_else(|| DemucsSpectralDecoderError::DimMismatch {
                stage: "dispatch_per_freq_bin ct overflow".into(),
                expected: 0,
                actual: channels,
            })?;

    // Build per-freq-bin input maps for batched dispatch.
    // Weights are shared across all freq bins via Cow::Borrowed to avoid
    // O(F × W) memory amplification from redundant clones.
    let shared_weights: Vec<(&str, Cow<'_, [f32]>)> = dconv_weights
        .iter()
        .map(|(name, w)| (name.as_str(), Cow::Borrowed(w.as_slice())))
        .collect();

    let mut batch_inputs: Vec<HashMap<&str, Cow<'_, [f32]>>> = Vec::with_capacity(freq_bins);
    for f in 0..freq_bins {
        // Extract [C, T] for this freq bin from [C, F, T] layout.
        // In row-major [C, F, T]: element(c, f, t) = c * F * T + f * T + t.
        let mut slice = Vec::with_capacity(ct);
        for c in 0..channels {
            let offset = c * freq_bins * time_len + f * time_len;
            slice.extend_from_slice(&data[offset..offset + time_len]);
        }

        let mut inputs: HashMap<&str, Cow<'_, [f32]>> =
            HashMap::with_capacity(1 + shared_weights.len());
        inputs.insert(nn_dsl::input_names::DATA, Cow::Owned(slice));
        for (name, w) in &shared_weights {
            inputs.insert(name, w.clone());
        }
        batch_inputs.push(inputs);
    }

    let outputs =
        execute_tensor_dispatch_batched(cache, dconv_def, ScalarType::F32, &batch_inputs)?;

    // outputs[f] is [C, T]. Transpose from [F, C, T] order to [C, F, T].
    let rearranged_cap =
        ct.checked_mul(freq_bins)
            .ok_or_else(|| DemucsSpectralDecoderError::DimMismatch {
                stage: "dispatch_per_freq_bin rearranged overflow".into(),
                expected: 0,
                actual: channels,
            })?;
    let mut rearranged = vec![0.0f32; rearranged_cap];
    for (f, result) in outputs.iter().enumerate() {
        for c in 0..channels {
            let src_offset = c * time_len;
            let dst_offset = c * freq_bins * time_len + f * time_len;
            rearranged[dst_offset..dst_offset + time_len]
                .copy_from_slice(&result[src_offset..src_offset + time_len]);
        }
    }
    Ok(rearranged)
}

/// Dispatch ConvTranspose1d independently for each time step.
///
/// Input: flattened `[C, F, T]`. For each of T time steps, extracts
/// `[C, F]` and dispatches ConvTranspose1d. Output: `[C_out, F_out, T]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_per_time_step(
    cache: &PipelineCache,
    conv_tr_def: &TensorKernelDef,
    data: &[f32],
    conv_tr_weights: &HashMap<String, Vec<f32>>,
    in_ch: usize,
    out_ch: usize,
    freq_in: usize,
    time_len: usize,
    target_f: usize,
) -> Result<Vec<f32>, DemucsSpectralDecoderError> {
    // Build per-time-step input maps for batched dispatch.
    // Weights are shared across all time steps via Cow::Borrowed to avoid
    // O(T × W) memory amplification from redundant clones.
    let shared_weights: Vec<(&str, Cow<'_, [f32]>)> = conv_tr_weights
        .iter()
        .map(|(name, w)| (name.as_str(), Cow::Borrowed(w.as_slice())))
        .collect();

    let mut batch_inputs: Vec<HashMap<&str, Cow<'_, [f32]>>> = Vec::with_capacity(time_len);
    for t in 0..time_len {
        // Extract [C, F] for this time step from [C, F, T] layout.
        // element(c, f, t) = c * F * T + f * T + t.
        let slice_cap =
            in_ch
                .checked_mul(freq_in)
                .ok_or_else(|| DemucsSpectralDecoderError::DimMismatch {
                    stage: "dispatch_per_time_step slice overflow".into(),
                    expected: 0,
                    actual: in_ch,
                })?;
        let mut slice = Vec::with_capacity(slice_cap);
        for c in 0..in_ch {
            for f in 0..freq_in {
                let idx = c * freq_in * time_len + f * time_len + t;
                slice.push(data[idx]);
            }
        }

        let mut inputs: HashMap<&str, Cow<'_, [f32]>> =
            HashMap::with_capacity(1 + shared_weights.len());
        inputs.insert(nn_dsl::input_names::DATA, Cow::Owned(slice));
        for (name, w) in &shared_weights {
            inputs.insert(name, w.clone());
        }
        batch_inputs.push(inputs);
    }

    let outputs =
        execute_tensor_dispatch_batched(cache, conv_tr_def, ScalarType::F32, &batch_inputs)?;

    // outputs[t] is [C_out, F_out]. Transpose to [C_out, F_out, T].
    let rearranged_cap = out_ch
        .checked_mul(target_f)
        .and_then(|v| v.checked_mul(time_len))
        .ok_or_else(|| DemucsSpectralDecoderError::DimMismatch {
            stage: "dispatch_per_time_step rearranged overflow".into(),
            expected: 0,
            actual: out_ch,
        })?;
    let mut rearranged = vec![0.0f32; rearranged_cap];
    for (t, result) in outputs.iter().enumerate() {
        for c in 0..out_ch {
            for f in 0..target_f {
                let src_idx = c * target_f + f;
                let dst_idx = c * target_f * time_len + f * time_len + t;
                rearranged[dst_idx] = result[src_idx];
            }
        }
    }
    Ok(rearranged)
}

/// GELU activation (tanh approximation, matching nn-dsl `gelu_eval`).
pub(super) fn gelu_f32(x: f32) -> f32 {
    let k: f32 = 0.797_884_6; // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    let e2 = (2.0 * inner).exp();
    0.5 * x * (2.0 - 2.0 / (e2 + 1.0))
}

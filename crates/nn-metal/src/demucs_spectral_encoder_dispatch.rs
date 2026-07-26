// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Axis-switch dispatch helpers for the Demucs spectral encoder.
//!
//! The spectral branch operates on 3D tensors `[C, F, T]` but nn's 1D ops
//! (Conv1d, DConv) work on `[C, N]`. These helpers extract slices along one
//! spatial axis, dispatch the 1D op, and reassemble the output.
//!
//! Part of #831 — spectral encoder.

use std::borrow::Cow;
use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::ScalarType;

use crate::tensor_dispatch::execute_tensor_dispatch_batched;
use crate::PipelineCache;

use super::DemucsSpectralEncoderError;

/// Dispatch Conv1d+GELU (or Rewrite+GLU) for each time step independently.
///
/// Input: flattened `[C_in, F_in, T]`. For each of T time steps, extracts
/// `[C_in, F_in]` and dispatches the sub-def. Output: `[C_out, F_out, T]`.
///
/// This handles both the main conv (F_in→F_out with downsampling) and the
/// rewrite conv (F preserved).
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_per_time_step(
    cache: &PipelineCache,
    def: &TensorKernelDef,
    data: &[f32],
    weights: &HashMap<String, Vec<f32>>,
    in_ch: usize,
    out_ch: usize,
    f_in: usize,
    f_out: usize,
    time_len: usize,
) -> Result<Vec<f32>, DemucsSpectralEncoderError> {
    // Build per-time-step input maps for batched dispatch.
    // Weights are shared across all time steps via Cow::Borrowed to avoid
    // O(T × W) memory amplification from redundant clones.
    let shared_weights: Vec<(&str, Cow<'_, [f32]>)> = weights
        .iter()
        .map(|(name, w)| (name.as_str(), Cow::Borrowed(w.as_slice())))
        .collect();

    let ft = f_in
        .checked_mul(time_len)
        .ok_or_else(|| DemucsSpectralEncoderError::DimMismatch {
            stage: "dispatch_per_time_step ft overflow".into(),
            expected: 0,
            actual: f_in,
        })?;
    let mut batch_inputs: Vec<HashMap<&str, Cow<'_, [f32]>>> = Vec::with_capacity(time_len);
    for t in 0..time_len {
        // Extract [C_in, F_in] for this time step from [C_in, F_in, T] layout.
        // For each channel, copy F_in elements at stride time_len using
        // contiguous row chunks instead of element-by-element indexing (#1357 AC5).
        let slice_cap =
            in_ch
                .checked_mul(f_in)
                .ok_or_else(|| DemucsSpectralEncoderError::DimMismatch {
                    stage: "dispatch_per_time_step slice overflow".into(),
                    expected: 0,
                    actual: in_ch,
                })?;
        let mut slice = Vec::with_capacity(slice_cap);
        for c in 0..in_ch {
            let row_base = c * ft + t;
            for f in 0..f_in {
                slice.push(data[row_base + f * time_len]);
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

    let outputs = execute_tensor_dispatch_batched(cache, def, ScalarType::F32, &batch_inputs)?;

    // outputs[t] is [C_out, F_out]. Reassemble into [C_out, F_out, T] layout
    // using strided writes (#1357 AC5).
    let ft_out =
        f_out
            .checked_mul(time_len)
            .ok_or_else(|| DemucsSpectralEncoderError::DimMismatch {
                stage: "dispatch_per_time_step ft_out overflow".into(),
                expected: 0,
                actual: f_out,
            })?;
    let rearranged_cap =
        out_ch
            .checked_mul(ft_out)
            .ok_or_else(|| DemucsSpectralEncoderError::DimMismatch {
                stage: "dispatch_per_time_step rearranged overflow".into(),
                expected: 0,
                actual: out_ch,
            })?;
    let mut rearranged = vec![0.0f32; rearranged_cap];
    for (t, result) in outputs.iter().enumerate() {
        for c in 0..out_ch {
            let dst_base = c * ft_out + t;
            let src_base = c * f_out;
            for f in 0..f_out {
                rearranged[dst_base + f * time_len] = result[src_base + f];
            }
        }
    }
    Ok(rearranged)
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
) -> Result<Vec<f32>, DemucsSpectralEncoderError> {
    let ct =
        channels
            .checked_mul(time_len)
            .ok_or_else(|| DemucsSpectralEncoderError::DimMismatch {
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
        // element(c, f, t) = c * F * T + f * T + t.
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
            .ok_or_else(|| DemucsSpectralEncoderError::DimMismatch {
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

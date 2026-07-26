// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Center-trim helpers for `DemucsTemporalDecoder`.
//!
//! Extracted from `demucs_temporal_decoder_forward.rs` (#2003) to keep files
//! under the 500-line limit.

use std::collections::HashMap;

use nn_core::DType;
use nn_dsl::ScalarType;

use crate::buffer::MetalBuffer;
use crate::gpu_slice::GpuSlice;
use crate::tensor_dispatch::{execute_tensor_dispatch_to_buffer, DispatchInput};
use crate::PipelineCache;

use crate::demucs_temporal_decoder::DemucsTemporalDecoderError;

/// Center-trim a GPU buffer representing `[C, T_current]` to `[C, target_t]`.
///
/// Uses `TensorBlockBuilder::add_narrow` on dim=1 (the T axis) to perform
/// the trim entirely on GPU. Returns a [`GpuSlice`] with the trimmed data.
pub(super) fn gpu_center_trim_1d(
    cache: &PipelineCache,
    skip_buf: &MetalBuffer,
    channels: usize,
    current_t: usize,
    target_t: usize,
) -> Result<GpuSlice, DemucsTemporalDecoderError> {
    let delta =
        current_t
            .checked_sub(target_t)
            .ok_or_else(|| DemucsTemporalDecoderError::DimMismatch {
                stage: "gpu_center_trim_1d: target > current".into(),
                expected: current_t,
                actual: target_t,
            })?;
    let start = delta / 2;
    let in_shape = &[channels, current_t];
    let out_shape = [channels, target_t];

    let def = crate::kernel_def_cache::get_or_build(
        "decoder_skip_trim",
        &[in_shape],
        &[start as u64, target_t as u64],
        DType::F32,
        || {
            let mut b = nn_dsl::TensorBlockBuilder::new("skip_center_trim");
            let input = b.add_input("skip_in", in_shape);
            let narrowed = b.add_narrow(input, 1, start, target_t, &out_shape);
            crate::build_kernel(b, narrowed)
        },
    )
    .map_err(|e| DemucsTemporalDecoderError::DimMismatch {
        stage: format!("gpu_center_trim kernel build: {e}"),
        expected: channels * current_t,
        actual: channels * target_t,
    })?;

    let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
    inputs.insert(
        "skip_in",
        DispatchInput::Gpu(GpuSlice::from_ref(skip_buf, 0)),
    );

    let slice = execute_tensor_dispatch_to_buffer::<f32>(cache, &def, ScalarType::F32, &inputs)?;
    Ok(slice)
}

/// Center-trim a flattened `[C, T_current]` tensor along the T dimension.
///
/// Returns `[C, target_t]`, taking the center `target_t` elements from each
/// channel row. Matches Python `center_trim`.
pub(in crate::demucs_temporal_decoder) fn center_trim_1d(
    data: &[f32],
    channels: usize,
    current_t: usize,
    target_t: usize,
) -> Result<Vec<f32>, DemucsTemporalDecoderError> {
    if current_t == target_t {
        return Ok(data.to_vec());
    }
    let delta =
        current_t
            .checked_sub(target_t)
            .ok_or_else(|| DemucsTemporalDecoderError::DimMismatch {
                stage: "center_trim_1d: target > current".into(),
                expected: current_t,
                actual: target_t,
            })?;
    let start = delta / 2;
    let cap =
        channels
            .checked_mul(target_t)
            .ok_or_else(|| DemucsTemporalDecoderError::DimMismatch {
                stage: "center_trim_1d allocation overflow".into(),
                expected: 0,
                actual: channels,
            })?;
    let mut out = Vec::with_capacity(cap);
    for c in 0..channels {
        let row_start = c * current_t + start;
        out.extend_from_slice(&data[row_start..row_start + target_t]);
    }
    Ok(out)
}

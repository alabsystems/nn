// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Forward pass implementation for `DemucsTemporalDecoder`.
//!
//! Extracted from `demucs_temporal_decoder.rs` (#1410) to keep files under the
//! 500-line limit. Contains the core inference loop: `forward_impl()` and the
//! `BlockDispatch` enum. Center-trim helpers live in `demucs_temporal_decoder_trim.rs`.

use std::collections::HashMap;

use nn_core::layers::{nan_check_policy, NanCheckPolicy};
use nn_dsl::ScalarType;

use crate::buffer::MetalBuffer;
use crate::element::MetalElement;
use crate::gpu_scope::with_gpu_scope;
use crate::gpu_slice::GpuSlice;
use crate::tensor_dispatch::{
    execute_tensor_dispatch, execute_tensor_dispatch_to_buffer, output_elems, DispatchInput,
};
use crate::{PipelineCache, TensorDispatchError};

use super::{DemucsTemporalDecoder, DemucsTemporalDecoderError, DEPTH};

#[path = "demucs_temporal_decoder_trim.rs"]
mod trim;
pub(super) use trim::center_trim_1d;
use trim::gpu_center_trim_1d;

impl DemucsTemporalDecoder {
    /// Run the temporal decoder forward pass (CPU dispatch).
    ///
    /// `bottleneck`: flattened `[channels_at_depth(DEPTH-1), T_bottleneck]`.
    /// `skips`: encoder skip connections in encoder order (depth 0..3),
    /// each flattened `[channels_at_depth(d), T_d]`.
    ///
    /// Returns flattened `[OUTPUT_CHANNELS, T_original]`.
    ///
    /// Skip connections are center-trimmed on CPU to match each block's
    /// temporal input dimension (matching Python HTDemucs `center_trim`).
    pub fn forward(
        &self,
        cache: &PipelineCache,
        bottleneck: &[f32],
        skips: &[Vec<f32>],
    ) -> Result<Vec<f32>, DemucsTemporalDecoderError> {
        self.forward_impl(cache, bottleneck, skips, None, BlockDispatch::Cpu)
    }

    /// GPU-optimized forward pass with buffer-to-buffer dispatch.
    ///
    /// Same semantics as [`forward`](Self::forward), but chains decoder blocks
    /// via `DispatchInput::Gpu` where possible. Weight buffers are persistent
    /// on GPU (uploaded once via `OnceLock`). The "data" input chains between
    /// blocks on GPU; skip connections are center-trimmed on CPU and uploaded
    /// per block as `DispatchInput::Cpu`.
    ///
    /// The savings come from eliminating per-dispatch CPU→GPU weight re-upload
    /// (~80 weight tensors across 4 blocks) and chaining the data input
    /// between blocks via GPU buffers.
    pub fn forward_gpu(
        &self,
        cache: &PipelineCache,
        bottleneck: &[f32],
        skips: &[Vec<f32>],
    ) -> Result<Vec<f32>, DemucsTemporalDecoderError> {
        self.forward_impl(cache, bottleneck, skips, None, BlockDispatch::Gpu)
    }

    /// GPU-optimized forward pass with GPU-resident skip connections.
    ///
    /// When GPU skip buffers are provided (from `EncoderOutput::skips_gpu`),
    /// center-trim is performed on GPU via narrow dispatch, eliminating
    /// 4 CPU→GPU skip re-uploads per forward pass. Falls back to CPU
    /// center-trim for any block where the GPU skip buffer is absent.
    pub fn forward_gpu_with_skips(
        &self,
        cache: &PipelineCache,
        bottleneck: &[f32],
        skips: &[Vec<f32>],
        skips_gpu: &[MetalBuffer],
    ) -> Result<Vec<f32>, DemucsTemporalDecoderError> {
        self.forward_impl(
            cache,
            bottleneck,
            skips,
            Some(skips_gpu),
            BlockDispatch::Gpu,
        )
    }

    fn forward_impl(
        &self,
        cache: &PipelineCache,
        bottleneck: &[f32],
        skips: &[Vec<f32>],
        skips_gpu: Option<&[MetalBuffer]>,
        dispatch: BlockDispatch,
    ) -> Result<Vec<f32>, DemucsTemporalDecoderError> {
        if skips.len() != DEPTH {
            return Err(DemucsTemporalDecoderError::DimMismatch {
                stage: "skips.len".to_string(),
                expected: DEPTH,
                actual: skips.len(),
            });
        }

        // Validate bottleneck length.
        let expected_bottleneck = self.block_in_ch[0] * self.block_t_in[0];
        if bottleneck.len() != expected_bottleneck {
            return Err(DemucsTemporalDecoderError::DimMismatch {
                stage: "bottleneck".to_string(),
                expected: expected_bottleneck,
                actual: bottleneck.len(),
            });
        }

        // When NanCheckPolicy::Skip is active and dispatching on GPU, wrap all
        // block dispatches in with_gpu_scope (#1985). This batches 4 per-block
        // commit_and_wait barriers into 1, since check_non_finite_err is a no-op
        // under Skip and f32::read_buffer can be deferred to after scope exit.
        let use_scoped =
            dispatch == BlockDispatch::Gpu && nan_check_policy() == NanCheckPolicy::Skip;

        if use_scoped {
            self.forward_gpu_scoped(cache, bottleneck, skips, skips_gpu)
        } else {
            self.forward_gpu_unscoped(cache, bottleneck, skips, skips_gpu, dispatch)
        }
    }

    /// GPU forward path with GpuScope batching (#1985).
    ///
    /// All 4 decoder block dispatches share a single `CommandBatch` via
    /// `with_gpu_scope`. No per-block `read_buffer` or NaN checks inside the
    /// scope. The final output is read back after the scope exits.
    ///
    /// Requires `NanCheckPolicy::Skip` — caller validates this.
    fn forward_gpu_scoped(
        &self,
        cache: &PipelineCache,
        bottleneck: &[f32],
        skips: &[Vec<f32>],
        skips_gpu: Option<&[MetalBuffer]>,
    ) -> Result<Vec<f32>, DemucsTemporalDecoderError> {
        let gpu_weights = self.ensure_gpu_weights(cache)?;

        // Pre-compute CPU center-trimmed skips for each block (needed for
        // DispatchInput::Cpu inside the scope when no GPU skip buffer exists).
        let mut trimmed_skips: Vec<Vec<f32>> = Vec::with_capacity(DEPTH);
        for block_idx in 0..DEPTH {
            let encoder_depth = DEPTH - 1 - block_idx;
            let in_ch = self.block_in_ch[block_idx];
            let t_in = self.block_t_in[block_idx];
            let skip_raw = &skips[encoder_depth];
            let skip_t = skip_raw.len() / in_ch;
            if skip_t < t_in {
                return Err(DemucsTemporalDecoderError::SkipTooShort {
                    depth: encoder_depth,
                    required: t_in,
                    actual: skip_t,
                });
            }
            trimmed_skips.push(center_trim_1d(skip_raw, in_ch, skip_t, t_in)?);
        }

        // Scope batches all 4 decoder dispatches into 1 commit_and_wait.
        let output_slice = with_gpu_scope(|| {
            let mut gpu_slice: Option<GpuSlice> = None;

            for block_idx in 0..DEPTH {
                let encoder_depth = DEPTH - 1 - block_idx;
                let in_ch = self.block_in_ch[block_idx];
                let t_in = self.block_t_in[block_idx];
                let skip_raw = &skips[encoder_depth];
                let skip_t = skip_raw.len() / in_ch;

                let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();

                // Chain from previous block's GPU slice, or upload bottleneck
                // for block 0.
                if let Some(slice) = gpu_slice.as_ref() {
                    inputs.insert(
                        nn_dsl::input_names::DATA,
                        DispatchInput::Gpu(slice.alias()),
                    );
                } else {
                    inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(bottleneck));
                }

                // GPU skip: center-trim on GPU if a GPU buffer is available,
                // otherwise use pre-computed CPU center-trimmed data.
                let gpu_skip_trimmed;
                if let Some(gpu_skips) = skips_gpu.filter(|s| encoder_depth < s.len()) {
                    if skip_t == t_in {
                        inputs.insert(
                            nn_dsl::input_names::SKIP,
                            DispatchInput::Gpu(GpuSlice::from_ref(&gpu_skips[encoder_depth], 0)),
                        );
                        gpu_skip_trimmed = None;
                    } else {
                        let trimmed_slice = gpu_center_trim_1d(
                            cache,
                            &gpu_skips[encoder_depth],
                            in_ch,
                            skip_t,
                            t_in,
                        )?;
                        gpu_skip_trimmed = Some(trimmed_slice);
                        let skip_ref = gpu_skip_trimmed.as_ref().ok_or_else(|| {
                            nn_core::TensorError::InvalidShape(
                                "gpu_skip_trimmed: unreachable None".to_string(),
                            )
                        })?;
                        inputs.insert(
                            nn_dsl::input_names::SKIP,
                            DispatchInput::Gpu(skip_ref.alias()),
                        );
                    }
                } else {
                    inputs.insert(
                        nn_dsl::input_names::SKIP,
                        DispatchInput::Cpu(&trimmed_skips[block_idx]),
                    );
                    gpu_skip_trimmed = None;
                }
                let _ = gpu_skip_trimmed;

                // Persistent GPU weight buffers (zero-copy alias).
                for (name, weight_buf) in &gpu_weights[block_idx] {
                    inputs.insert(
                        name.as_str(),
                        DispatchInput::Gpu(GpuSlice::from_ref(weight_buf, 0)),
                    );
                }

                let slice = execute_tensor_dispatch_to_buffer::<f32>(
                    cache,
                    &self.block_defs[block_idx],
                    ScalarType::F32,
                    &inputs,
                )?;
                gpu_slice = Some(slice);
            }

            // Return the final block's GPU buffer for readback after scope.
            gpu_slice.ok_or_else(|| {
                nn_core::TensorError::InvalidShape("decoder: no blocks dispatched".to_string())
            })
        })
        .map_err(|te| {
            DemucsTemporalDecoderError::Dispatch(TensorDispatchError::Metal(
                crate::error::MetalError::DispatchFailed(format!("gpu_scope: {te}")),
            ))
        })?;

        // Scope exited — data is committed, read_buffer_at_offset is safe.
        let last_block = DEPTH - 1;
        let elems = output_elems(
            &self.block_defs[last_block],
            self.block_defs[last_block].output,
        )?;
        let data =
            f32::read_buffer_at_offset(output_slice.buffer(), output_slice.byte_offset(), elems)
                .map_err(TensorDispatchError::from)?;
        Ok(data)
    }

    /// Unscoped GPU/CPU forward path (original behavior).
    ///
    /// Used when `NanCheckPolicy::Skip` is NOT active, or for CPU dispatch.
    /// Each block dispatches independently with per-block `read_buffer` and
    /// NaN/Inf validation.
    fn forward_gpu_unscoped(
        &self,
        cache: &PipelineCache,
        bottleneck: &[f32],
        skips: &[Vec<f32>],
        skips_gpu: Option<&[MetalBuffer]>,
        dispatch: BlockDispatch,
    ) -> Result<Vec<f32>, DemucsTemporalDecoderError> {
        let mut data = bottleneck.to_vec();
        // GPU mode: track the previous GpuSlice for buffer-to-buffer chaining.
        let mut gpu_slice: Option<GpuSlice> = None;

        for block_idx in 0..DEPTH {
            let encoder_depth = DEPTH - 1 - block_idx;
            let in_ch = self.block_in_ch[block_idx];
            let t_in = self.block_t_in[block_idx];
            let skip_raw = &skips[encoder_depth];

            // Validate no NaN/Inf in data (defense-in-depth, single-pass count).
            crate::check_non_finite_err(&data, |count| {
                DemucsTemporalDecoderError::NonFiniteInput {
                    block: block_idx,
                    count,
                }
            })?;

            // Center-trim skip to match current temporal dimension.
            let skip_t = skip_raw.len() / in_ch;
            if skip_t < t_in {
                return Err(DemucsTemporalDecoderError::SkipTooShort {
                    depth: encoder_depth,
                    required: t_in,
                    actual: skip_t,
                });
            }
            let trimmed_skip = center_trim_1d(skip_raw, in_ch, skip_t, t_in)?;

            match dispatch {
                BlockDispatch::Cpu => {
                    let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
                    inputs.insert(nn_dsl::input_names::DATA, &data);
                    inputs.insert(nn_dsl::input_names::SKIP, &trimmed_skip);
                    for (name, weight_data) in &self.block_weights[block_idx] {
                        inputs.insert(name.as_str(), weight_data.as_slice());
                    }
                    data = execute_tensor_dispatch(
                        cache,
                        &self.block_defs[block_idx],
                        ScalarType::F32,
                        &inputs,
                    )?;
                }
                BlockDispatch::Gpu => {
                    let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
                    // Chain data from previous block's GPU slice if available.
                    if let Some(slice) = gpu_slice.as_ref() {
                        inputs.insert(
                            nn_dsl::input_names::DATA,
                            DispatchInput::Gpu(slice.alias()),
                        );
                    } else {
                        inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(&data));
                    }
                    // GPU skip: center-trim on GPU if a GPU buffer is available,
                    // eliminating one CPU→GPU re-upload per decoder block.
                    let gpu_skip_trimmed;
                    if let Some(gpu_skips) = skips_gpu.filter(|s| encoder_depth < s.len()) {
                        if skip_t == t_in {
                            // No trim needed — use the GPU buffer directly.
                            inputs.insert(
                                nn_dsl::input_names::SKIP,
                                DispatchInput::Gpu(GpuSlice::from_ref(
                                    &gpu_skips[encoder_depth],
                                    0,
                                )),
                            );
                            gpu_skip_trimmed = None;
                        } else {
                            // Center-trim on GPU via narrow dispatch.
                            let trimmed_slice = gpu_center_trim_1d(
                                cache,
                                &gpu_skips[encoder_depth],
                                in_ch,
                                skip_t,
                                t_in,
                            )?;
                            gpu_skip_trimmed = Some(trimmed_slice);
                            // gpu_skip_trimmed is guaranteed Some after the line above.
                            // ok_or avoids a panic path even though it's unreachable.
                            let skip_ref = gpu_skip_trimmed.as_ref().ok_or_else(|| {
                                DemucsTemporalDecoderError::DimMismatch {
                                    stage: "gpu_skip_trimmed".to_string(),
                                    expected: 1,
                                    actual: 0,
                                }
                            })?;
                            inputs.insert(
                                nn_dsl::input_names::SKIP,
                                DispatchInput::Gpu(skip_ref.alias()),
                            );
                        }
                    } else {
                        // No GPU skips — use CPU center-trimmed data.
                        inputs.insert(
                            nn_dsl::input_names::SKIP,
                            DispatchInput::Cpu(&trimmed_skip),
                        );
                        gpu_skip_trimmed = None;
                    }
                    let _ = gpu_skip_trimmed;
                    // Persistent GPU weight buffers (zero-copy alias).
                    let gpu_weights = self.ensure_gpu_weights(cache)?;
                    for (name, weight_buf) in &gpu_weights[block_idx] {
                        inputs.insert(
                            name.as_str(),
                            DispatchInput::Gpu(GpuSlice::from_ref(weight_buf, 0)),
                        );
                    }
                    let slice = execute_tensor_dispatch_to_buffer::<f32>(
                        cache,
                        &self.block_defs[block_idx],
                        ScalarType::F32,
                        &inputs,
                    )?;
                    // Read back for NaN/Inf check on next block and final output.
                    // Use offset-aware readback for arena-allocated buffers.
                    let elems = output_elems(
                        &self.block_defs[block_idx],
                        self.block_defs[block_idx].output,
                    )?;
                    data = f32::read_buffer_at_offset(slice.buffer(), slice.byte_offset(), elems)
                        .map_err(TensorDispatchError::from)?;
                    gpu_slice = Some(slice);
                }
            }
        }

        Ok(data)
    }
}

/// Controls whether decoder blocks use GPU buffer-to-buffer dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockDispatch {
    /// CPU round-trip dispatch between decoder blocks.
    Cpu,
    /// Buffer-to-buffer GPU dispatch between decoder blocks.
    Gpu,
}

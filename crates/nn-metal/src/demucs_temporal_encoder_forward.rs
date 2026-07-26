// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Forward pass implementation for `DemucsTemporalEncoder`.
//!
//! Extracted from `demucs_temporal_encoder.rs` (#1410) to keep files under the
//! 500-line limit. Contains the core inference loop: `forward_impl()`,
//! `forward_debug()`, and supporting types (`DebugCollector`, `BlockDispatch`).

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

use super::{
    DemucsTemporalEncoder, DemucsTemporalEncoderError, EncoderOutput, AUDIO_CHANNELS, DEPTH, STRIDE,
};

impl DemucsTemporalEncoder {
    /// Run the temporal encoder forward pass.
    ///
    /// `audio`: flattened `[AUDIO_CHANNELS, T]` waveform.
    ///
    /// Returns `EncoderOutput` containing the bottleneck, skip connections
    /// (in encoder depth order), and input lengths at each depth.
    pub fn forward(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
    ) -> Result<EncoderOutput, DemucsTemporalEncoderError> {
        self.forward_impl(cache, audio, BlockDispatch::Cpu, None)
    }

    /// GPU-optimized forward pass with buffer-to-buffer dispatch.
    ///
    /// Same semantics as [`forward`](Self::forward), but chains encoder blocks
    /// via `DispatchInput::Gpu` where possible. When a block requires stride
    /// padding (CPU-only operation), the GPU buffer is read back, padded, and
    /// re-uploaded. For typical audio lengths (stride-aligned), all 4 blocks
    /// chain on GPU without intermediate readback.
    ///
    /// Skip connections and the bottleneck are still returned as CPU `Vec<f32>`
    /// because the decoder's `center_trim` and the transformer's `transpose`
    /// operate on CPU. The savings come from eliminating up to 3 GPU->CPU->GPU
    /// round-trips between encoder blocks.
    pub fn forward_gpu(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
    ) -> Result<EncoderOutput, DemucsTemporalEncoderError> {
        self.forward_impl(cache, audio, BlockDispatch::Gpu, None)
    }

    fn forward_impl(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        dispatch: BlockDispatch,
        debug: Option<&mut DebugCollector>,
    ) -> Result<EncoderOutput, DemucsTemporalEncoderError> {
        // Validate audio is channel-aligned.
        if !audio.len().is_multiple_of(AUDIO_CHANNELS) {
            return Err(DemucsTemporalEncoderError::DimMismatch {
                stage: "audio_input".to_string(),
                expected: 0, // signals "must be multiple of AUDIO_CHANNELS"
                actual: audio.len() % AUDIO_CHANNELS,
            });
        }

        // Validate audio length matches the initial_t used at construction.
        if dispatch == BlockDispatch::Gpu {
            let in_ch_0 = AUDIO_CHANNELS;
            let t_in_0 = self.block_t_in[0];
            let expected_len = in_ch_0 * t_in_0;
            if audio.len() != expected_len {
                return Err(DemucsTemporalEncoderError::DimMismatch {
                    stage: "audio_length".to_string(),
                    expected: expected_len,
                    actual: audio.len(),
                });
            }
        }

        // When NanCheckPolicy::Skip is active and dispatching on GPU, wrap all
        // block dispatches in with_gpu_scope (#1985). This batches 4 per-block
        // commit_and_wait barriers into 1, since check_non_finite_err is a no-op
        // under Skip and f32::read_buffer can be deferred to after scope exit.
        let use_scoped =
            dispatch == BlockDispatch::Gpu && nan_check_policy() == NanCheckPolicy::Skip;

        if use_scoped {
            self.forward_gpu_scoped(cache, audio)
        } else {
            self.forward_gpu_unscoped(cache, audio, dispatch, debug)
        }
    }

    /// GPU forward path with GpuScope batching (#1985).
    ///
    /// All 4 encoder block dispatches share a single `CommandBatch` via
    /// `with_gpu_scope`. No per-block `read_buffer` or NaN checks inside the
    /// scope. Skip buffers and bottleneck are read back after the scope exits.
    ///
    /// Requires `NanCheckPolicy::Skip` — caller validates this.
    fn forward_gpu_scoped(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
    ) -> Result<EncoderOutput, DemucsTemporalEncoderError> {
        // Pre-compute input lengths from the block_t_in set at construction.
        let input_lengths: Vec<usize> = self.block_t_in.clone();
        let gpu_weights = self.ensure_gpu_weights(cache)?;

        // Scope batches all 4 encoder dispatches into 1 commit_and_wait.
        let skips_gpu = with_gpu_scope(|| {
            let mut gpu_slice: Option<GpuSlice> = None;
            let mut scope_skips: Vec<GpuSlice> = Vec::with_capacity(DEPTH);

            for block_idx in 0..DEPTH {
                let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();

                // Chain from previous block's GPU slice, or upload audio for block 0.
                if let Some(slice) = gpu_slice.as_ref() {
                    inputs.insert(
                        nn_dsl::input_names::DATA,
                        DispatchInput::Gpu(slice.alias()),
                    );
                } else {
                    inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(audio));
                }

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

                scope_skips.push(slice.alias());
                gpu_slice = Some(slice);
            }

            Ok(scope_skips)
        })
        .map_err(|te| {
            DemucsTemporalEncoderError::Dispatch(TensorDispatchError::Metal(
                crate::error::MetalError::DispatchFailed(format!("gpu_scope: {te}")),
            ))
        })?;

        // Scope exited — data is committed, read_buffer_at_offset is safe.
        let mut skips = Vec::with_capacity(DEPTH);
        let mut skips_bufs = Vec::with_capacity(DEPTH);
        for (block_idx, skip_slice) in skips_gpu.iter().enumerate() {
            let out_ch = self.block_out_ch[block_idx];
            let elems = output_elems(
                &self.block_defs[block_idx],
                self.block_defs[block_idx].output,
            )?;
            let data =
                f32::read_buffer_at_offset(skip_slice.buffer(), skip_slice.byte_offset(), elems)
                    .map_err(TensorDispatchError::from)?;
            if !data.len().is_multiple_of(out_ch) {
                return Err(DemucsTemporalEncoderError::OutputAlignment {
                    block: block_idx,
                    actual: data.len(),
                    channels: out_ch,
                });
            }
            skips_bufs.push(skip_slice.buffer().alias());
            skips.push(data);
        }

        // Bottleneck is the last skip connection's data.
        let bottleneck = skips.last().cloned().unwrap_or_default();

        Ok(EncoderOutput {
            bottleneck,
            skips,
            skips_gpu: Some(skips_bufs),
            input_lengths,
        })
    }

    /// Unscoped GPU/CPU forward path (original behavior).
    ///
    /// Used when `NanCheckPolicy::Skip` is NOT active, or for CPU dispatch.
    /// Each block dispatches independently with per-block `read_buffer` and
    /// NaN/Inf validation.
    fn forward_gpu_unscoped(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        dispatch: BlockDispatch,
        mut debug: Option<&mut DebugCollector>,
    ) -> Result<EncoderOutput, DemucsTemporalEncoderError> {
        let mut data = audio.to_vec();
        let mut skips = Vec::with_capacity(DEPTH);
        let mut skips_gpu: Vec<MetalBuffer> = Vec::with_capacity(DEPTH);
        let mut input_lengths = Vec::with_capacity(DEPTH);
        // GPU mode: track the previous GpuSlice for buffer-to-buffer chaining.
        let mut gpu_slice: Option<GpuSlice> = None;

        for block_idx in 0..DEPTH {
            let out_ch = self.block_out_ch[block_idx];
            let in_ch = if block_idx == 0 {
                AUDIO_CHANNELS
            } else {
                self.block_out_ch[block_idx - 1]
            };

            // Validate no NaN/Inf in data (defense-in-depth, single-pass count).
            crate::check_non_finite_err(&data, |count| {
                DemucsTemporalEncoderError::NonFiniteInput {
                    block: block_idx,
                    count,
                }
            })?;

            // Record input length before padding (decoder needs this for trim).
            let t_in = data.len() / in_ch;
            input_lengths.push(t_in);

            // Pad to stride multiple on CPU (matching Python HEncLayer).
            let needs_pad = !t_in.is_multiple_of(STRIDE);
            if needs_pad {
                let pad_per_channel = STRIDE - (t_in % STRIDE);
                let padded_t = t_in + pad_per_channel;
                let mut padded = vec![0.0f32; in_ch * padded_t];
                for c in 0..in_ch {
                    let src_start = c * t_in;
                    let dst_start = c * padded_t;
                    padded[dst_start..dst_start + t_in]
                        .copy_from_slice(&data[src_start..src_start + t_in]);
                }
                data = padded;
            }

            // Capture block input for debug trace.
            if let Some(ref mut dbg) = debug {
                dbg.block_inputs.push(data.clone());
            }

            // Dispatch block through CPU or GPU path.
            match dispatch {
                BlockDispatch::Cpu => {
                    let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
                    inputs.insert(nn_dsl::input_names::DATA, &data);
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
                    // Use GPU buffer if available and no stride padding was needed
                    // (stride padding consumes `data` on CPU, invalidating the GPU buffer).
                    let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
                    if let (Some(slice), false) = (gpu_slice.as_ref(), needs_pad) {
                        inputs.insert(
                            nn_dsl::input_names::DATA,
                            DispatchInput::Gpu(slice.alias()),
                        );
                    } else {
                        inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(&data));
                    }
                    // Use persistent GPU weight buffers (zero-copy alias) instead
                    // of re-uploading CPU data on every dispatch call.
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
                    // Keep GPU buffer for skip connection (decoder can use DispatchInput::Gpu).
                    skips_gpu.push(slice.buffer().alias());
                    // Read back for NaN/Inf check and CPU skip (backward compat).
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

            // Skip connection: output of this block.
            if !data.len().is_multiple_of(out_ch) {
                return Err(DemucsTemporalEncoderError::OutputAlignment {
                    block: block_idx,
                    actual: data.len(),
                    channels: out_ch,
                });
            }

            // Capture block output for debug trace.
            if let Some(ref mut dbg) = debug {
                dbg.block_outputs.push(data.clone());
            }

            skips.push(data.clone());

            // data flows to next block as input.
        }

        let skips_gpu = if dispatch == BlockDispatch::Gpu && !skips_gpu.is_empty() {
            Some(skips_gpu)
        } else {
            None
        };
        Ok(EncoderOutput {
            bottleneck: data,
            skips,
            skips_gpu,
            input_lengths,
        })
    }

    /// Run the temporal encoder forward pass with per-block intermediate dumps.
    ///
    /// Returns `(EncoderOutput, block_inputs, block_outputs)` where `block_inputs[i]`
    /// is the data fed to block `i` (after stride padding) and `block_outputs[i]` is
    /// the output of block `i`.
    ///
    /// Used for debugging parity failures by comparing intermediates against Python.
    pub fn forward_debug(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
    ) -> Result<(EncoderOutput, Vec<Vec<f32>>, Vec<Vec<f32>>), DemucsTemporalEncoderError> {
        let mut dbg = DebugCollector::default();
        let output = self.forward_impl(cache, audio, BlockDispatch::Cpu, Some(&mut dbg))?;
        Ok((output, dbg.block_inputs, dbg.block_outputs))
    }
}

/// Collects per-block intermediate tensors for debugging parity failures.
///
/// Used by `forward_debug` to capture block inputs and outputs without
/// duplicating the entire encoder loop.
#[derive(Default)]
pub(super) struct DebugCollector {
    pub(super) block_inputs: Vec<Vec<f32>>,
    pub(super) block_outputs: Vec<Vec<f32>>,
}

/// Controls whether encoder blocks use GPU buffer-to-buffer dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlockDispatch {
    /// CPU round-trip dispatch between encoder blocks.
    Cpu,
    /// Buffer-to-buffer GPU dispatch between encoder blocks.
    Gpu,
}

// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Forward pass methods for [`SileroVad`].
//!
//! Contains the core `forward()` pipeline (STFT → encoder → LSTM → output),
//! `forward_gpu()` (buffer-to-buffer GPU dispatch in the encoder), and
//! encoder dispatch helpers (CPU + GPU).
//!
//! Batch inference (`get_probabilities`, `get_speech_segments`) lives in
//! `silero_vad_batch.rs`.
//!
//! Part of #1065 (extraction) and #934 (CPU/GPU unification).

use std::collections::HashMap;

use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_dsl::ir::ScalarType;

use crate::cache::PipelineCache;
use crate::element::MetalElement;
use crate::gpu_slice::GpuSlice;
use crate::stft::compute_stft_magnitude;
use crate::tensor_dispatch::{
    execute_tensor_dispatch, execute_tensor_dispatch_to_buffer, output_elems, DispatchInput,
};

use super::error::SileroVadError;
use super::state::{
    SileroVadOutput, SileroVadState, AUDIO_CONTEXT_SIZE, CHUNK_SIZE, LSTM_HIDDEN_SIZE,
};
use super::validate_state_finiteness;

/// Controls whether the encoder stages use GPU buffer-to-buffer dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadDispatch {
    /// CPU round-trip dispatch between encoder blocks.
    Cpu,
    /// Buffer-to-buffer GPU dispatch between encoder blocks.
    Gpu,
}

impl super::SileroVad {
    /// Run a single forward pass on one audio chunk with streaming state.
    ///
    /// The model internally prepends `state.context` (64 samples from the
    /// previous chunk) to the 512 new samples to form the 576-sample STFT
    /// input, matching PyTorch's internal context handling.
    ///
    /// For streaming VAD, pass the returned `SileroVadOutput.state` into the
    /// next call. For single-chunk (stateless) usage, pass `SileroVadState::zero()`.
    ///
    /// # Arguments
    ///
    /// * `cache` — Metal pipeline cache for kernel compilation.
    /// * `audio` — 512-sample audio chunk (new samples only).
    /// * `state` — Streaming state from the previous chunk (or `SileroVadState::zero()`).
    ///
    /// # Returns
    ///
    /// `SileroVadOutput` containing the speech probability and updated state
    /// (LSTM h/c + audio context for the next chunk).
    pub fn forward(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        state: &SileroVadState,
    ) -> Result<SileroVadOutput, SileroVadError> {
        self.forward_impl(cache, audio, state, VadDispatch::Cpu)
    }

    /// GPU-optimized forward pass with buffer-to-buffer dispatch.
    ///
    /// Same semantics as [`forward`](Self::forward), but keeps encoder
    /// intermediate results on GPU between stages. Only the LSTM output
    /// (needed for streaming state) and final probability are read back
    /// to CPU.
    ///
    /// **Performance:** Eliminates 4 of 6 CPU<->GPU round-trips for the
    /// encoder chain. LSTM and output stages still read back because their
    /// results feed into the returned `SileroVadState`.
    pub fn forward_gpu(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        state: &SileroVadState,
    ) -> Result<SileroVadOutput, SileroVadError> {
        // NaN-skip scope: per-stage encoder checks are no-ops during GPU dispatch.
        // Model-boundary checks (input audio, LSTM state, output probability) use
        // direct is_finite()/validate_state_finiteness() — unaffected by the policy.
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            self.forward_impl(cache, audio, state, VadDispatch::Gpu)
        })
    }

    /// Process one chunk using GPU-optimized dispatch, updating state in place.
    ///
    /// Convenience wrapper around [`forward_gpu`](Self::forward_gpu), matching
    /// the [`process`](Self::process) API pattern.
    pub fn process_gpu(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        state: &mut SileroVadState,
    ) -> Result<f32, SileroVadError> {
        let out = self.forward_gpu(cache, audio, state)?;
        *state = out.state;
        Ok(out.probability)
    }

    fn forward_impl(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        state: &SileroVadState,
        dispatch: VadDispatch,
    ) -> Result<SileroVadOutput, SileroVadError> {
        if audio.len() != CHUNK_SIZE {
            return Err(SileroVadError::AudioLength {
                expected: CHUNK_SIZE,
                actual: audio.len(),
            });
        }
        // Validate streaming state dimensions — wrong-size state produces wrong
        // STFT input or corrupts GPU dispatch buffers silently.
        if state.context.len() != AUDIO_CONTEXT_SIZE {
            return Err(SileroVadError::StateDimension {
                field: "context",
                expected: AUDIO_CONTEXT_SIZE,
                actual: state.context.len(),
            });
        }
        if state.h_state.len() != LSTM_HIDDEN_SIZE {
            return Err(SileroVadError::StateDimension {
                field: "h_state",
                expected: LSTM_HIDDEN_SIZE,
                actual: state.h_state.len(),
            });
        }
        if state.c_state.len() != LSTM_HIDDEN_SIZE {
            return Err(SileroVadError::StateDimension {
                field: "c_state",
                expected: LSTM_HIDDEN_SIZE,
                actual: state.c_state.len(),
            });
        }

        // Validate incoming state finiteness — SileroVadState has public fields,
        // so callers can construct states with NaN/Inf. Catch at the model boundary
        // to prevent corrupted state from silently poisoning LSTM computation and
        // all subsequent chunks (AC2, #929).
        validate_state_finiteness(state)?;

        // Reject NaN/Inf audio — these silently propagate through STFT and produce
        // garbage probabilities. Validate at the model boundary (defense-in-depth).
        let first_bad = audio.iter().position(|v| !v.is_finite());
        if let Some(first_index) = first_bad {
            let count = audio.iter().filter(|v| !v.is_finite()).count();
            return Err(SileroVadError::NonFiniteAudio { count, first_index });
        }
        let dtype = ScalarType::F32;

        // Step 0: Prepend context from previous chunk to form 576-sample STFT input.
        // context[64] + audio[512] = stft_input[576].
        let mut stft_input = Vec::with_capacity(AUDIO_CONTEXT_SIZE + CHUNK_SIZE);
        stft_input.extend_from_slice(&state.context);
        stft_input.extend_from_slice(audio);

        // Step 1: CPU STFT → [129, 4] magnitude spectrogram.
        let stft_mag = compute_stft_magnitude(&stft_input, &self.stft_basis, &self.stft_params)?;

        // Step 2: Encoder blocks 0-3 — dispatch via CPU or GPU buffer-to-buffer.
        let data = match dispatch {
            VadDispatch::Cpu => self.dispatch_encoder_cpu(cache, stft_mag, dtype)?,
            VadDispatch::Gpu => self.dispatch_encoder_gpu(cache, &stft_mag, dtype)?,
        };

        // Step 3: Squeeze — [1, 128, 1] flattened is 128 floats = [1, 128]. No-op.

        // Step 4: LSTM cell with caller-provided state.
        // Dual-output def returns [h_new; c_new] as a flat [2*128] = 256-float buffer.
        // GPU path uses persistent weight buffers; CPU path re-uploads each call.
        let lstm_out = match dispatch {
            VadDispatch::Gpu => {
                let gw = self.ensure_gpu_weights(cache)?;
                let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
                inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(&data));
                inputs.insert(
                    nn_dsl::input_names::HIDDEN_STATE,
                    DispatchInput::Cpu(&state.h_state),
                );
                inputs.insert(
                    nn_dsl::input_names::CELL_STATE,
                    DispatchInput::Cpu(&state.c_state),
                );
                inputs.insert(
                    nn_dsl::input_names::WEIGHT_IH,
                    DispatchInput::Gpu(GpuSlice::from_ref(&gw.lstm_weight_ih, 0)),
                );
                inputs.insert(
                    nn_dsl::input_names::WEIGHT_HH,
                    DispatchInput::Gpu(GpuSlice::from_ref(&gw.lstm_weight_hh, 0)),
                );
                inputs.insert(
                    nn_dsl::input_names::BIAS,
                    DispatchInput::Gpu(GpuSlice::from_ref(&gw.lstm_bias, 0)),
                );
                let slice = execute_tensor_dispatch_to_buffer::<f32>(
                    cache,
                    &self.lstm_def,
                    dtype,
                    &inputs,
                )?;
                let elems = output_elems(&self.lstm_def, self.lstm_def.output)?;
                f32::read_buffer_at_offset(slice.buffer(), slice.byte_offset(), elems)
                    .map_err(crate::tensor_dispatch::TensorDispatchError::from)?
            }
            VadDispatch::Cpu => {
                let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
                inputs.insert(nn_dsl::input_names::DATA, &data);
                inputs.insert(nn_dsl::input_names::HIDDEN_STATE, &state.h_state);
                inputs.insert(nn_dsl::input_names::CELL_STATE, &state.c_state);
                inputs.insert(nn_dsl::input_names::WEIGHT_IH, &self.lstm_weight_ih);
                inputs.insert(nn_dsl::input_names::WEIGHT_HH, &self.lstm_weight_hh);
                inputs.insert(nn_dsl::input_names::BIAS, &self.lstm_bias);
                execute_tensor_dispatch(cache, &self.lstm_def, dtype, &inputs)?
            }
        };

        // Split stacked output: first 128 = h_new, last 128 = c_new.
        let expected_lstm_len = 2 * LSTM_HIDDEN_SIZE;
        if lstm_out.len() != expected_lstm_len {
            return Err(SileroVadError::OutputLength {
                stage: "lstm",
                expected: expected_lstm_len,
                actual: lstm_out.len(),
            });
        }
        let h_new = lstm_out[..LSTM_HIDDEN_SIZE].to_vec();
        let c_new = lstm_out[LSTM_HIDDEN_SIZE..].to_vec();

        // Defense-in-depth: catch NaN/Inf in LSTM state before returning it.
        // NaN in h_new/c_new silently propagates across streaming chunks,
        // corrupting all subsequent inference (AC1, #929).
        let count = crate::count_non_finite(&h_new) + crate::count_non_finite(&c_new);
        if count > 0 {
            return Err(SileroVadError::NonFiniteLstmState { count });
        }

        // Step 5: Output stage (ReLU + Linear + Sigmoid).
        // GPU path uses persistent weight buffers; CPU path re-uploads each call.
        let prob = match dispatch {
            VadDispatch::Gpu => {
                let gw = self.ensure_gpu_weights(cache)?;
                let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
                inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(&h_new));
                inputs.insert(
                    nn_dsl::input_names::WEIGHT,
                    DispatchInput::Gpu(GpuSlice::from_ref(&gw.output_weight, 0)),
                );
                inputs.insert(
                    nn_dsl::input_names::BIAS,
                    DispatchInput::Gpu(GpuSlice::from_ref(&gw.output_bias, 0)),
                );
                let slice = execute_tensor_dispatch_to_buffer::<f32>(
                    cache,
                    &self.output_def,
                    dtype,
                    &inputs,
                )?;
                let elems = output_elems(&self.output_def, self.output_def.output)?;
                f32::read_buffer_at_offset(slice.buffer(), slice.byte_offset(), elems)
                    .map_err(crate::tensor_dispatch::TensorDispatchError::from)?
            }
            VadDispatch::Cpu => {
                let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
                inputs.insert(nn_dsl::input_names::DATA, &h_new);
                inputs.insert(nn_dsl::input_names::WEIGHT, &self.output_weight);
                inputs.insert(nn_dsl::input_names::BIAS, &self.output_bias);
                execute_tensor_dispatch(cache, &self.output_def, dtype, &inputs)?
            }
        };

        // Extract probability — output stage produces exactly 1 float.
        let probability = *prob.first().ok_or(SileroVadError::OutputLength {
            stage: "output",
            expected: 1,
            actual: 0,
        })?;

        // Defense-in-depth: reject NaN/Inf output before returning to consumer.
        if !probability.is_finite() {
            return Err(SileroVadError::NonFiniteOutput { value: probability });
        }

        // Save the last 64 samples of the new audio as context for the next chunk.
        let new_context = audio[audio.len() - AUDIO_CONTEXT_SIZE..].to_vec();

        Ok(SileroVadOutput {
            probability,
            state: SileroVadState {
                h_state: h_new,
                c_state: c_new,
                context: new_context,
            },
        })
    }

    /// CPU encoder dispatch: each block reads back to CPU between stages.
    fn dispatch_encoder_cpu(
        &self,
        cache: &PipelineCache,
        stft_mag: Vec<f32>,
        dtype: ScalarType,
    ) -> Result<Vec<f32>, SileroVadError> {
        let mut data = stft_mag;
        for i in 0..4 {
            let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
            inputs.insert(nn_dsl::input_names::DATA, &data);
            inputs.insert(nn_dsl::input_names::WEIGHT, &self.enc_weights[i]);
            inputs.insert(nn_dsl::input_names::BIAS, &self.enc_biases[i]);
            data = execute_tensor_dispatch(cache, &self.enc_defs[i], dtype, &inputs)?;

            // Defense-in-depth: catch NaN/Inf between encoder stages before they
            // propagate through remaining blocks and LSTM (AC2, #789).
            crate::check_non_finite_err(&data, |count| SileroVadError::NonFiniteEncoder {
                block: i,
                count,
            })?;
        }
        Ok(data)
    }

    /// GPU encoder dispatch: chains blocks via Metal buffers, reads back once.
    ///
    /// Uses persistent GPU weight buffers (lazily initialized on first call)
    /// to avoid re-uploading encoder weights/biases on every forward pass.
    ///
    /// All 4 encoder block dispatches share a single [`with_gpu_scope`] command
    /// buffer (#1854), reducing 4 `commit_and_wait` barriers to 1. The scope
    /// commits when the closure returns; `read_buffer` after the scope sees
    /// committed data.
    fn dispatch_encoder_gpu(
        &self,
        cache: &PipelineCache,
        stft_mag: &[f32],
        dtype: ScalarType,
    ) -> Result<Vec<f32>, SileroVadError> {
        // Lazily upload all weights to GPU once; subsequent calls use cached buffers.
        let gw = self.ensure_gpu_weights(cache)?;

        // Scope batches all 4 encoder dispatches into 1 commit_and_wait.
        let enc_slice = crate::gpu_scope::with_gpu_scope(|| {
            // Block 0: CPU data input → GPU buffer, GPU weights (persistent).
            let mut enc0_inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
            enc0_inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(stft_mag));
            enc0_inputs.insert(
                nn_dsl::input_names::WEIGHT,
                DispatchInput::Gpu(GpuSlice::from_ref(&gw.enc[0].0, 0)),
            );
            enc0_inputs.insert(
                nn_dsl::input_names::BIAS,
                DispatchInput::Gpu(GpuSlice::from_ref(&gw.enc[0].1, 0)),
            );
            let mut enc_slice = execute_tensor_dispatch_to_buffer::<f32>(
                cache,
                &self.enc_defs[0],
                dtype,
                &enc0_inputs,
            )
            .map_err(nn_core::TensorError::from)?;

            // Blocks 1-3: GPU buffer -> GPU buffer, GPU weights (persistent).
            for i in 1..4 {
                let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
                inputs.insert(
                    nn_dsl::input_names::DATA,
                    DispatchInput::Gpu(enc_slice.alias()),
                );
                inputs.insert(
                    nn_dsl::input_names::WEIGHT,
                    DispatchInput::Gpu(GpuSlice::from_ref(&gw.enc[i].0, 0)),
                );
                inputs.insert(
                    nn_dsl::input_names::BIAS,
                    DispatchInput::Gpu(GpuSlice::from_ref(&gw.enc[i].1, 0)),
                );
                enc_slice = execute_tensor_dispatch_to_buffer::<f32>(
                    cache,
                    &self.enc_defs[i],
                    dtype,
                    &inputs,
                )
                .map_err(nn_core::TensorError::from)?;
            }

            Ok(enc_slice)
        })
        .map_err(|te| {
            SileroVadError::Dispatch(crate::tensor_dispatch::TensorDispatchError::Metal(
                crate::error::MetalError::DispatchFailed(format!("encoder gpu_scope: {te}")),
            ))
        })?;

        // Read back final encoder output for LSTM input (128 floats).
        // Happens after scope exit -- commit_and_wait already completed.
        // Use offset-aware readback for arena-allocated buffers (#2207).
        let last_enc = &self.enc_defs[3];
        let enc_elems = output_elems(last_enc, last_enc.output)?;
        let data: Vec<f32> =
            f32::read_buffer_at_offset(enc_slice.buffer(), enc_slice.byte_offset(), enc_elems)
                .map_err(crate::tensor_dispatch::TensorDispatchError::from)?;

        // Defense-in-depth: check encoder output for NaN/Inf.
        crate::check_non_finite_err(&data, |count| SileroVadError::NonFiniteEncoder {
            block: 3,
            count,
        })?;

        Ok(data)
    }
}

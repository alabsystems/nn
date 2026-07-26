// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Silero VAD model execution on nn Metal backend.
//!
//! First end-to-end model running entirely through nn's verified pipeline.
//! Phase 1: CPU round-trip between dispatch calls. Each encoder block, LSTM,
//! and output stage is a separate `TensorKernelDef` dispatched on Metal.
//!
//! Architecture (16kHz, 512-sample chunks):
//! ```text
//! Audio [1, 576] → STFT (CPU) → [1, 129, 4]
//!   → Encoder 0: Conv1d(129→128, k=3, s=1, p=1) + ReLU → [1, 128, 4]
//!   → Encoder 1: Conv1d(128→64,  k=3, s=2, p=1) + ReLU → [1, 64, 2]
//!   → Encoder 2: Conv1d(64→64,   k=3, s=2, p=1) + ReLU → [1, 64, 1]
//!   → Encoder 3: Conv1d(64→128,  k=3, s=1, p=1) + ReLU → [1, 128, 1]
//!   → Squeeze(dim=2): [1, 128]
//!   → LSTM cell → h_new [1, 128]
//!   → ReLU → Linear(128→1) → Sigmoid → speech probability
//! ```
//!
//! Part of #761 — Direction 5 (model struct + forward pass).

use crate::{GpuWeightCache, GpuWeightRef};
use std::path::Path;

use nn_dsl::lstm_decomposed::build_lstm_cell_decomposed_dual;
use nn_dsl::TensorKernelDef;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::element::MetalElement;
use crate::stft::StftParams;

// Builder and validation helpers extracted (Part of #761).
#[path = "silero_vad_builders.rs"]
mod builders;
use builders::{
    build_encoder_block_def, build_output_def, conv1d_output_len, validate_all_weights,
    ENCODER_BLOCKS,
};

// Weight struct and safetensors loading extracted (Part of #886).
#[path = "silero_vad_weights.rs"]
mod weights;
pub use weights::SileroVadWeights;

// Error types extracted (Part of #892).
#[path = "silero_vad_error.rs"]
mod error;
pub use error::SileroVadError;

// Streaming state and output types extracted (Part of #894).
#[path = "silero_vad_state.rs"]
mod state;
pub use state::{SileroVadOutput, SileroVadState};
use state::{AUDIO_CONTEXT_SIZE, CHUNK_SIZE, LSTM_HIDDEN_SIZE};
// Speech segment detection (ported from dvoice-vad segments.rs).
#[path = "silero_vad_segments.rs"]
pub(crate) mod segments;
pub use segments::{SegmentConfig, SpeechSegment};

// Forward pass (CPU + GPU) and encoder dispatch helpers.
#[path = "silero_vad_forward.rs"]
mod forward;

// Batch inference: get_probabilities() and get_speech_segments().
#[path = "silero_vad_batch.rs"]
mod batch;

/// Validate incoming streaming state for finiteness (AC2, #929).
///
/// `SileroVadState` has public `Vec<f32>` fields, so callers can construct
/// states with NaN/Inf directly. This check catches corrupted state at the
/// model boundary before it poisons LSTM computation.
fn validate_state_finiteness(state: &SileroVadState) -> Result<(), SileroVadError> {
    for (field, data) in [
        ("h_state", state.h_state.as_slice()),
        ("c_state", state.c_state.as_slice()),
        ("context", state.context.as_slice()),
    ] {
        let count = data.iter().filter(|v| !v.is_finite()).count();
        if count > 0 {
            return Err(SileroVadError::NonFiniteInputState { field, count });
        }
    }
    Ok(())
}

/// Holds all weight tensors as persistent Metal GPU buffers.
/// Lazily initialized on first `forward_gpu()` call via `OnceLock`.
struct VadGpuWeights {
    /// Encoder weight/bias pairs for blocks 0-3.
    enc: [(MetalBuffer, MetalBuffer); 4],
    /// LSTM: weight_ih, weight_hh, bias.
    lstm_weight_ih: MetalBuffer,
    lstm_weight_hh: MetalBuffer,
    lstm_bias: MetalBuffer,
    /// Output: weight, bias.
    output_weight: MetalBuffer,
    output_bias: MetalBuffer,
}

/// Silero VAD 16kHz model — Phase 1 (CPU round-trip dispatch).
///
/// Holds pre-built `TensorKernelDef`s for each stage and CPU-side weight data.
/// Forward pass chains Metal dispatch calls with CPU-side intermediate buffers.
#[must_use]
pub struct SileroVad {
    stft_params: StftParams,
    stft_basis: Vec<f32>,
    enc_defs: [TensorKernelDef; 4],
    lstm_def: TensorKernelDef,
    output_def: TensorKernelDef,
    enc_weights: [Vec<f32>; 4],
    enc_biases: [Vec<f32>; 4],
    lstm_weight_ih: Vec<f32>,
    lstm_weight_hh: Vec<f32>,
    /// Combined LSTM bias: bias_ih + bias_hh (element-wise sum).
    lstm_bias: Vec<f32>,
    output_weight: Vec<f32>,
    output_bias: Vec<f32>,
    /// Lazily-initialized GPU buffers for all model weights.
    /// Populated on first `forward_gpu()` call, eliminating per-dispatch
    /// CPU→GPU re-upload (~13 weight tensors per forward).
    gpu_weights: GpuWeightCache<VadGpuWeights>,
}

impl std::fmt::Debug for SileroVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_params: usize = self.stft_basis.len()
            + self.enc_weights.iter().map(Vec::len).sum::<usize>()
            + self.enc_biases.iter().map(Vec::len).sum::<usize>()
            + self.lstm_weight_ih.len()
            + self.lstm_weight_hh.len()
            + self.lstm_bias.len()
            + self.output_weight.len()
            + self.output_bias.len();
        f.debug_struct("SileroVad")
            .field("total_params", &total_params)
            .field("encoder_blocks", &4)
            .finish_non_exhaustive()
    }
}
impl SileroVad {
    /// Create a new Silero VAD model from weight tensors.
    ///
    /// The two LSTM biases (`bias_ih`, `bias_hh`) are combined by element-wise
    /// addition, matching PyTorch's `LSTMCell` semantics where both biases are
    /// additive: `gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh`.
    ///
    /// # Errors
    ///
    /// Returns `SileroVadError::WeightSize` if any weight tensor has the wrong
    /// number of elements.
    pub fn new(weights: SileroVadWeights) -> Result<Self, SileroVadError> {
        validate_all_weights(&weights)?;

        // Combine LSTM biases: bias = bias_ih + bias_hh (element-wise).
        let lstm_bias: Vec<f32> = weights
            .lstm_bias_ih
            .iter()
            .zip(&weights.lstm_bias_hh)
            .map(|(a, b)| a + b)
            .collect();

        // Validate combined bias finiteness — individual biases near f32::MAX/2
        // can overflow to infinity when summed (AC1, #789).
        if let Some(first_index) = lstm_bias.iter().position(|v| !v.is_finite()) {
            let count = lstm_bias.iter().filter(|v| !v.is_finite()).count();
            return Err(SileroVadError::NonFiniteBias { count, first_index });
        }

        // Compute temporal dimensions through encoder.
        let stft_params = StftParams::default();
        let stft_input_len = AUDIO_CONTEXT_SIZE + CHUNK_SIZE; // 576
        let stft_t = conv1d_output_len(
            stft_input_len + stft_params.pad_right,
            stft_params.n_fft,
            stft_params.hop_length,
            0,
        )?;
        // stft_t = (640 - 256) / 128 + 1 = 4

        // Encoder temporal progression: 4 → 4 → 2 → 1 → 1
        let mut enc_t = [0usize; 4];
        let mut t = stft_t;
        for (i, blk) in ENCODER_BLOCKS.iter().enumerate() {
            t = conv1d_output_len(t, blk.kernel_size, blk.stride, blk.padding)?;
            enc_t[i] = t;
        }

        let mut t_in = stft_t;
        let mut enc_defs_vec = Vec::with_capacity(4);
        for i in 0..4 {
            let t_out = enc_t[i];
            enc_defs_vec.push(build_encoder_block_def(&ENCODER_BLOCKS[i], t_in, t_out)?);
            t_in = t_out;
        }
        let enc_defs: [TensorKernelDef; 4] =
            enc_defs_vec
                .try_into()
                .map_err(|v: Vec<_>| SileroVadError::OutputLength {
                    stage: "encoder_defs",
                    expected: 4,
                    actual: v.len(),
                })?;

        let lstm_def =
            build_lstm_cell_decomposed_dual(LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE, 1, true)?;
        let output_def = build_output_def()?;

        Ok(Self {
            stft_params,
            stft_basis: weights.stft_basis,
            enc_defs,
            lstm_def,
            output_def,
            enc_weights: weights.enc_weights,
            enc_biases: weights.enc_biases,
            lstm_weight_ih: weights.lstm_weight_ih,
            lstm_weight_hh: weights.lstm_weight_hh,
            lstm_bias,
            output_weight: weights.output_weight,
            output_bias: weights.output_bias,
            gpu_weights: GpuWeightCache::new(),
        })
    }

    /// Lazily upload all model weights to persistent GPU buffers.
    ///
    /// Called on the first `forward_gpu()` invocation. Subsequent calls return
    /// the cached buffers (via `OnceLock`). Each weight `Vec<f32>` is uploaded
    /// once via `f32::create_buffer`, and the resulting `MetalBuffer` is reused
    /// for all future dispatches as `DispatchInput::Gpu` (zero-copy alias).
    /// Covers encoder (4 weight + 4 bias), LSTM (3), and output (2) = 13 total.
    fn ensure_gpu_weights(
        &self,
        cache: &PipelineCache,
    ) -> Result<GpuWeightRef<'_, VadGpuWeights>, SileroVadError> {
        self.gpu_weights.get_or_init_with(
            || {
                let ctx = cache.context();
                let upload = |data: &[f32]| -> Result<MetalBuffer, String> {
                    f32::create_buffer(ctx, data).map_err(|e| e.to_string())
                };
                Ok(VadGpuWeights {
                    enc: [
                        (upload(&self.enc_weights[0])?, upload(&self.enc_biases[0])?),
                        (upload(&self.enc_weights[1])?, upload(&self.enc_biases[1])?),
                        (upload(&self.enc_weights[2])?, upload(&self.enc_biases[2])?),
                        (upload(&self.enc_weights[3])?, upload(&self.enc_biases[3])?),
                    ],
                    lstm_weight_ih: upload(&self.lstm_weight_ih)?,
                    lstm_weight_hh: upload(&self.lstm_weight_hh)?,
                    lstm_bias: upload(&self.lstm_bias)?,
                    output_weight: upload(&self.output_weight)?,
                    output_bias: upload(&self.output_bias)?,
                })
            },
            SileroVadError::GpuBufferAlloc,
        )
    }

    /// Load a Silero VAD model from a safetensors file (15 tensors).
    ///
    /// Combines [`WeightMap::load_global`] + [`SileroVadWeights::from_weight_map`]
    /// + [`SileroVad::new`]. Requires [`MetalBackend::init`](crate::MetalBackend::init).
    ///
    /// # Safety
    ///
    /// The file must not be modified during loading. After return the file is
    /// no longer referenced (weights are copied into owned buffers).
    pub unsafe fn load(path: impl AsRef<Path>) -> Result<Self, SileroVadError> {
        // SAFETY: Caller guarantees the file is not modified during loading
        // (forwarded from this function's own `# Safety` contract).
        let wm = unsafe { crate::safetensors::WeightMap::load_global(path.as_ref())? };
        let weights = SileroVadWeights::from_weight_map(&wm)?;
        Self::new(weights)
    }

    /// Load from a safetensors file without mmap (fully safe, no `unsafe`).
    ///
    /// Reads the file into memory and extracts tensors. For zero-copy
    /// loading with Metal shared buffers, use [`load`](Self::load) instead.
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Self, SileroVadError> {
        let weights = SileroVadWeights::from_safetensors_file(path)?;
        Self::new(weights)
    }

    /// Process one 512-sample audio chunk, updating streaming state in place.
    ///
    /// Convenience wrapper around [`forward`](Self::forward) matching
    /// dvoice's `process(&mut self, &[f32]) -> Result<f32>` pattern.
    /// Mutates `state` with the updated LSTM h/c and audio context.
    pub fn process(
        &self,
        cache: &PipelineCache,
        audio: &[f32],
        state: &mut SileroVadState,
    ) -> Result<f32, SileroVadError> {
        let out = self.forward(cache, audio, state)?;
        *state = out.state;
        Ok(out.probability)
    }
}

// Re-import for test submodules that use `super::super::*` to access
// symbols from this module scope (e.g., silero_vad_lstm_numeric_tests.rs).
#[cfg(test)]
use crate::tensor_dispatch::execute_tensor_dispatch;
#[cfg(test)]
use nn_dsl::ir::ScalarType;

#[cfg(test)]
#[path = "silero_vad_tests.rs"]
mod tests;

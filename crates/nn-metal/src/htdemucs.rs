// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unified HTDemucs model: dual-branch (temporal + spectral) encoder →
//! transformer → decoder → iSTFT combine.
//!
//! Temporal-only mode: spectral weights `None` → zeros to transformer.
//! Dual-branch mode: `forward_with_stft()` with STFT `[4, F, T]` input.
//! Spectral decoder output is reconstructed via iSTFT and summed with
//! the temporal decoder output.
//!
//! Forward pass methods: `htdemucs_forward.rs`
//!
//! Part of #779, #831, and #961.

use std::path::Path;

use crate::demucs_spectral_decoder::DemucsSpectralDecoder;
use crate::demucs_spectral_encoder::DemucsSpectralEncoder;
use crate::demucs_temporal_decoder::DemucsTemporalDecoder;
use crate::demucs_temporal_encoder::DemucsTemporalEncoder;
use crate::demucs_transformer::DemucsTransformer;
use crate::istft::IstftBasis;
use crate::istft_gpu::IstftGpuBasis;

pub(crate) use crate::demucs_spectral_decoder::DemucsSpectralDecoderWeights;
pub(crate) use crate::demucs_spectral_encoder::DemucsSpectralEncoderWeights;
pub(crate) use crate::demucs_temporal_decoder::DemucsTemporalDecoderWeights;
pub(crate) use crate::demucs_temporal_encoder::DemucsTemporalEncoderWeights;
pub(crate) use crate::demucs_transformer::DemucsTransformerWeights;

#[path = "htdemucs_weights.rs"]
mod weights;
pub use weights::WeightLoadError;

#[path = "htdemucs_helpers.rs"]
mod helpers;
use helpers::{
    compute_bottleneck_t, compute_encoder_input_lengths, compute_spectral_encoder_freqs,
    denormalize_output, normalize_audio, spectral_reconstruct, spectral_reconstruct_gpu,
};

#[path = "htdemucs_error.rs"]
mod error;
pub use error::HTDemucsError;

#[path = "htdemucs_debug.rs"]
mod debug;

#[path = "htdemucs_construction.rs"]
mod construction;

#[cfg(test)]
#[path = "htdemucs_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "htdemucs_normalization_tests.rs"]
mod normalization_tests;

#[cfg(test)]
#[path = "htdemucs_spectral_reconstruct_tests.rs"]
mod spectral_reconstruct_tests;

#[cfg(all(test, feature = "bench"))]
#[path = "htdemucs_bench.rs"]
mod bench;

// ---------------------------------------------------------------------------
// Configuration constants (htdemucs_ft defaults)
// ---------------------------------------------------------------------------

/// Number of audio channels (stereo).
const AUDIO_CHANNELS: usize = 2;

/// Number of source outputs (vocals, drums, bass, other).
const NUM_SOURCES: usize = 4;

/// Output channels: NUM_SOURCES * AUDIO_CHANNELS.
const OUTPUT_CHANNELS: usize = NUM_SOURCES * AUDIO_CHANNELS;

/// Bottleneck channel dimension: channels_at_depth(3) = 48 * 2^3 = 384.
const BOTTLENECK_DIM: usize = 384;

// ---------------------------------------------------------------------------
// Weight type
// ---------------------------------------------------------------------------

/// All weights for the unified HTDemucs model.
///
/// Composes encoder, transformer, and decoder weight types.
/// Spectral branch weights are optional: when `None`, the model runs in
/// temporal-only mode (feeding zeros to the transformer's spectral branch).
#[derive(Debug, Clone)]
#[must_use = "HTDemucsWeights is a data transfer type; pass to HTDemucs::new()"]
#[non_exhaustive]
pub struct HTDemucsWeights {
    /// Temporal encoder weights (4 blocks).
    pub encoder: DemucsTemporalEncoderWeights,
    /// Transformer bottleneck weights (5 layers × 2 branches).
    pub transformer: DemucsTransformerWeights,
    /// Temporal decoder weights (4 blocks).
    pub decoder: DemucsTemporalDecoderWeights,
    /// Spectral encoder weights (optional, enables spectral branch).
    pub spectral_encoder: Option<DemucsSpectralEncoderWeights>,
    /// Spectral decoder weights (optional, enables spectral branch).
    pub spectral_decoder: Option<DemucsSpectralDecoderWeights>,
}

impl HTDemucsWeights {
    /// Create weights from sub-component weight structs.
    ///
    /// Set `spectral_encoder` and `spectral_decoder` to `None` for
    /// temporal-only mode, or `Some(...)` to enable the spectral branch.
    pub fn new(
        encoder: DemucsTemporalEncoderWeights,
        transformer: DemucsTransformerWeights,
        decoder: DemucsTemporalDecoderWeights,
        spectral_encoder: Option<DemucsSpectralEncoderWeights>,
        spectral_decoder: Option<DemucsSpectralDecoderWeights>,
    ) -> Self {
        Self {
            encoder,
            transformer,
            decoder,
            spectral_encoder,
            spectral_decoder,
        }
    }
}

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Unified HTDemucs: temporal + spectral branches → transformer → decoders → iSTFT combine.
#[must_use = "HTDemucs is constructed once and reused; call .forward() to run inference"]
pub struct HTDemucs {
    encoder: DemucsTemporalEncoder,
    transformer: DemucsTransformer,
    decoder: DemucsTemporalDecoder,
    /// Spectral encoder (None = temporal-only mode).
    spectral_encoder: Option<DemucsSpectralEncoder>,
    /// Spectral decoder (None = temporal-only mode).
    spectral_decoder: Option<DemucsSpectralDecoder>,
    /// Pre-computed iSTFT basis for spectral→waveform reconstruction (None = temporal-only).
    istft_basis: Option<IstftBasis>,
    /// Pre-uploaded GPU iSTFT basis (None = temporal-only or GPU init failed).
    istft_gpu_basis: Option<IstftGpuBasis>,
    /// Input audio temporal length.
    audio_t: usize,
    /// Transformer spectral sequence length (1 for temporal-only,
    /// or F_bottleneck * T for spectral mode).
    spectral_seq_len: usize,
    /// Expected STFT magnitude input length (when spectral branch is present).
    stft_expected_len: usize,
    /// STFT frequency dimension (n_bins = n_fft/2+1) when spectral branch present.
    stft_f: usize,
    /// STFT time dimension when spectral branch present.
    stft_t: usize,
}

impl std::fmt::Debug for HTDemucs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HTDemucs")
            .field("audio_t", &self.audio_t)
            .field("encoder", &self.encoder)
            .field("transformer", &self.transformer)
            .field("decoder", &self.decoder)
            .field("has_spectral", &self.spectral_encoder.is_some())
            .finish_non_exhaustive()
    }
}

/// Spectral encoder input channels: 2 (stereo) × 2 (real+imag) = 4.
const SPECTRAL_INPUT_CHANNELS: usize = 4;

/// Spectral bottleneck frequency dimension after 4 encoder blocks.
/// With F_in=2048 and stride=4 at each depth: 2048→512→128→32→8.
const SPECTRAL_BOTTLENECK_F: usize = 8;

impl HTDemucs {
    /// Construct in temporal-only mode (no spectral branch).
    pub fn new(weights: HTDemucsWeights, audio_t: usize) -> Result<Self, HTDemucsError> {
        Self::new_inner(weights, audio_t, None, None)
    }

    /// Construct with spectral branch enabled (requires `stft_f` and `stft_t`).
    pub fn new_with_spectral(
        weights: HTDemucsWeights,
        audio_t: usize,
        stft_f: usize,
        stft_t: usize,
    ) -> Result<Self, HTDemucsError> {
        Self::new_inner(weights, audio_t, Some(stft_f), Some(stft_t))
    }

    /// Load from safetensors via mmap. Requires `MetalBackend::init`.
    ///
    /// # Safety
    /// File must not be modified during loading.
    pub unsafe fn load(path: impl AsRef<Path>, audio_t: usize) -> Result<Self, HTDemucsError> {
        // SAFETY: Caller guarantees the file is not modified during loading
        // (forwarded from this function's own `# Safety` contract).
        let wm = unsafe {
            crate::safetensors::WeightMap::load_global(path.as_ref())
                .map_err(WeightLoadError::from)?
        };
        let weights = HTDemucsWeights::from_weight_map(&wm)?;
        Self::new(weights, audio_t)
    }

    /// Load from safetensors without mmap (fully safe, no Metal context needed).
    pub fn load_safetensors(path: impl AsRef<Path>, audio_t: usize) -> Result<Self, HTDemucsError> {
        let weights = HTDemucsWeights::from_safetensors_file(path)?;
        Self::new(weights, audio_t)
    }

    /// Minimum audio temporal dimension for HTDemucs.
    ///
    /// With 4 encoder blocks of Conv1d(kernel=8, stride=4, padding=2), the
    /// computation `ceil_to_stride(t) + 2*padding - kernel` underflows `usize`
    /// when `t == 0`. Require at least 1 sample per channel.
    const MINIMUM_AUDIO_T: usize = 1;

    /// Audio temporal length this model was built for.
    pub fn audio_t(&self) -> usize {
        self.audio_t
    }

    /// Whether the spectral branch is enabled.
    pub fn has_spectral(&self) -> bool {
        self.spectral_encoder.is_some()
    }
}

/// Check a buffer for NaN/Inf values, returning `NonFiniteIntermediate` on failure.
fn check_finite(data: &[f32], stage: &'static str) -> Result<(), HTDemucsError> {
    crate::check_non_finite_err(data, |count| HTDemucsError::NonFiniteIntermediate {
        stage,
        count,
    })
}

/// Controls whether the temporal encoder uses GPU buffer-to-buffer dispatch.
///
/// When `Gpu`, encoder blocks chain via Metal buffers (avoiding CPU round-trips).
/// When `Cpu`, encoder blocks use CPU dispatch (simpler, works without Metal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderDispatch {
    /// CPU round-trip dispatch between encoder blocks.
    Cpu,
    /// Buffer-to-buffer GPU dispatch between encoder blocks.
    Gpu,
}

// Forward pass methods extracted to htdemucs_forward.rs
#[path = "htdemucs_forward.rs"]
mod forward;

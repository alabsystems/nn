// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper STT model for nn.
//!
//! Backend-agnostic implementation using `DynTensor` + `Module` API.
//! Matches the interface of `candle_transformers::models::whisper::model::Whisper`.
//!
//! # Architecture
//!
//! - **Audio encoder:** Conv1d stem + sinusoidal positional + N transformer blocks + LayerNorm
//! - **Text decoder:** Token embedding + learned positional + N blocks (self+cross attn) + LayerNorm + tied linear
//!
//! # Usage
//!
//! ```no_run
//! use nn_whisper::{WhisperModel, WhisperConfig};
//! use nn_core::VarBuilder;
//!
//! let config = WhisperConfig::large_v3_turbo();
//! let vb = VarBuilder::zeros(nn_core::DType::F32, &nn_core::Device::Cpu);
//! let mut model = WhisperModel::load(&vb, config).expect("load model");
//! ```

#[doc(hidden)]
pub mod attention;
pub mod audio;
pub mod audio_processing;
#[doc(hidden)]
pub mod block;
pub mod config;
pub mod decode;
pub mod decoder;
pub mod encoder;
pub mod error;
#[doc(hidden)]
pub mod positional;
pub mod quality;
pub mod quality_audio;
#[cfg(any(test, feature = "test-helpers"))]
pub mod test_utils;
pub mod beam_search;
pub mod streaming;
pub mod tokenizer;

pub use audio::{
    mel_filterbank, pcm_to_mel, whisper_mel_spectrogram, whisper_mel_spectrogram_for_config,
};
pub use audio_processing::{normalize_audio, pad_or_trim, preprocess_audio, resample, stereo_to_mono};
pub use config::{
    WhisperConfig, CHUNK_LENGTH, HOP_LENGTH, NUM_MEL_BINS, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE,
};
pub use decode::{
    beam_search_decode, compression_ratio, decode_with_temperature, detect_language, greedy_decode,
    passes_quality_check, temperature_fallback_decode, transcribe, transcribe_long,
    transcribe_with_fallback, DecodeConfig, DecodingResult, LanguageDetectionResult,
    LongFormConfig, LongFormResult, LongFormSegment, TranscriptionResult, WhisperBeamConfig,
    DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_TEMPERATURES,
    MAX_DECODE_LENGTH,
};
pub use decoder::TextDecoder;
pub use encoder::AudioEncoder;
pub use error::WhisperError;
pub use streaming::{StreamingConfig, StreamingSegment, StreamingTranscriber};
pub use quality::{character_error_rate, match_error_rate, normalized_edit_distance, word_error_rate};
pub use quality_audio::{audio_snr, pesq_approximation};
pub use tokenizer::{
    DecodedSegment, WhisperTokenizer, DEFAULT_NO_SPEECH_THRESHOLD, EOT_TOKEN, LANGUAGE_TOKEN_END,
    LANGUAGE_TOKEN_START, NO_SPEECH_TOKEN, SOT_TOKEN,
};

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{check_output_finite, with_nan_check_policy, NanCheckPolicy};
use nn_core::{DType, Device, Result, TensorError, VarBuilder};
use std::path::Path;

/// Whisper speech-to-text model.
///
/// Provides `encode()` (mel → audio features) and `decode()` (tokens + audio → logits)
/// methods matching the candle Whisper interface.
pub struct WhisperModel {
    encoder: AudioEncoder,
    decoder: TextDecoder,
    config: WhisperConfig,
    /// Model weight dtype (from VarBuilder). Used to convert mel spectrogram
    /// input to match encoder weight dtype for bf16/f16 inference (#1721).
    dtype: DType,
}

impl WhisperModel {
    /// Load Whisper model from VarBuilder.
    ///
    /// VarBuilder should point to the model root. Weight keys are expected
    /// under `model.encoder.*` and `model.decoder.*` prefixes.
    pub fn load(vb: impl AsRef<VarBuilder>, config: WhisperConfig) -> Result<Self> {
        let vb = vb.as_ref();
        config.validate()?;
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        let dtype = vb.dtype();
        Ok(Self {
            encoder,
            decoder,
            config,
            dtype,
        })
    }

    /// Load Whisper model from a safetensors file.
    ///
    /// Reads the file into memory and parses all tensors into a `VarBuilder`.
    /// Weight keys are expected under `model.encoder.*` and `model.decoder.*`.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IoError` on read failure, `WhisperError::WeightLoad`
    /// on safetensors parse failure, or config validation errors.
    pub fn load_safetensors(path: impl AsRef<Path>, config: WhisperConfig) -> Result<Self> {
        let vb = load_safetensors_vb(path)?;
        Self::load(&vb, config)
    }

    /// Encode mel spectrogram to audio features.
    ///
    /// Input: `[batch, num_mel_bins, n_frames]` (e.g., `[1, 128, 3000]`)
    /// Output: `[batch, seq_len, d_model]` (e.g., `[1, 1500, 1280]`)
    ///
    /// The mel tensor is converted to the model's weight dtype before encoding.
    /// This handles the common case where `whisper_mel_spectrogram()` returns F32
    /// but the model was loaded with BF16/F16 weights (#1721).
    pub fn encode(&mut self, mel: &DynTensor) -> Result<DynTensor> {
        let mel = if mel.dtype() != self.dtype {
            mel.to_dtype(self.dtype)?
        } else {
            mel.clone()
        };
        // Skip per-block check_output_finite inside encoder — eliminates
        // N+1 GPU→CPU readback flushes (N blocks + 1 final). Boundary
        // check below catches any NaN from the full encoder pass.
        let output = with_nan_check_policy(NanCheckPolicy::Skip, || self.encoder.forward(&mel))?;
        check_output_finite(&output, "WhisperEncoder")?;
        Ok(output)
    }

    /// Decode one step: token IDs + encoder output → logits.
    ///
    /// - `tokens`: `[batch, seq_len]` U32 token IDs
    /// - `encoder_output`: from `encode()`
    /// - `flush_kv_cache`: **must be `true` on the first decode step of each
    ///   new audio segment.** Clears both self-attention and cross-attention
    ///   KV caches. Passing a different `encoder_output` without flushing
    ///   will return an error (stale cache detection).
    /// - `position_offset`: cumulative token position for positional embedding
    ///
    /// Returns: `[batch, seq_len, vocab_size]` logits.
    pub fn decode(
        &mut self,
        tokens: &DynTensor,
        encoder_output: &DynTensor,
        flush_kv_cache: bool,
        position_offset: usize,
    ) -> Result<DynTensor> {
        // Skip per-block check_output_finite inside decoder — eliminates
        // N+1 GPU→CPU readback flushes per decode step. Boundary check
        // below + decode loop's check_logit_finiteness provide defense-in-depth.
        let logits = with_nan_check_policy(NanCheckPolicy::Skip, || {
            self.decoder
                .forward(tokens, encoder_output, flush_kv_cache, position_offset)
        })?;
        check_output_finite(&logits, "WhisperDecoder")?;
        Ok(logits)
    }

    /// Reset all KV caches (call between utterances).
    ///
    /// Resets both decoder caches (self-attention + cross-attention) and
    /// encoder self-attention caches. The encoder also resets its own caches
    /// at the start of each `encode()` call, but this provides explicit
    /// cleanup for callers that need it.
    pub fn reset_kv_cache(&mut self) {
        self.encoder.reset_cache();
        self.decoder.reset_cache();
    }

    /// Model configuration.
    #[must_use]
    pub fn config(&self) -> &WhisperConfig {
        &self.config
    }

    /// Model weight dtype (from VarBuilder at load time).
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Borrow the encoder for direct access (e.g., tracing via `forward_no_cache`).
    #[cfg(any(test, feature = "test-helpers"))]
    #[must_use]
    pub fn encoder(&self) -> &AudioEncoder {
        &self.encoder
    }

    /// Borrow the decoder for direct access (e.g., tracing via `forward_no_cache`).
    #[cfg(any(test, feature = "test-helpers"))]
    #[must_use]
    pub fn decoder(&self) -> &TextDecoder {
        &self.decoder
    }
}

/// Convert raw safetensors bytes to f32 based on dtype.
///
/// Returns `Ok(Some(data))` for float types, `Ok(None)` for non-float (skip),
/// or `Err` on alignment issues.
fn convert_tensor_bytes(
    name: &str,
    bytes: &[u8],
    dtype: safetensors::Dtype,
) -> Result<Option<Vec<f32>>> {
    match dtype {
        safetensors::Dtype::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(WhisperError::ByteAlignment {
                    tensor_name: name.into(),
                    byte_len: bytes.len(),
                    alignment: 4,
                }
                .into());
            }
            Ok(Some(
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ))
        }
        safetensors::Dtype::BF16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(WhisperError::ByteAlignment {
                    tensor_name: name.into(),
                    byte_len: bytes.len(),
                    alignment: 2,
                }
                .into());
            }
            Ok(Some(
                bytes
                    .chunks_exact(2)
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                    .collect(),
            ))
        }
        safetensors::Dtype::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(WhisperError::ByteAlignment {
                    tensor_name: name.into(),
                    byte_len: bytes.len(),
                    alignment: 2,
                }
                .into());
            }
            Ok(Some(
                bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
            ))
        }
        _other => Ok(None),
    }
}

/// Load safetensors file into a CPU `VarBuilder`.
///
/// Parses the file, extracts all F32/BF16/F16 tensors, and builds an in-memory
/// `VarBuilder` using `TensorMapBackend`. BF16/F16 tensors are converted to F32.
fn load_safetensors_vb(path: impl AsRef<Path>) -> Result<VarBuilder> {
    let data = std::fs::read(path.as_ref()).map_err(TensorError::IoError)?;
    let st = safetensors::SafeTensors::deserialize(&data).map_err(|e| {
        TensorError::from(WhisperError::SafetensorsParseError {
            detail: e.to_string(),
        })
    })?;

    let mut tensors = std::collections::HashMap::new();
    for (name, view) in st.tensors() {
        let float_data = match convert_tensor_bytes(&name, view.data(), view.dtype())? {
            Some(d) => d,
            None => continue, // Skip non-float tensors (e.g., metadata).
        };

        // Validate finiteness at load time (#943 pattern).
        let non_finite = float_data.iter().filter(|v| !v.is_finite()).count();
        if non_finite > 0 {
            return Err(WhisperError::NonFiniteWeight {
                tensor_name: name,
                count: non_finite,
            }
            .into());
        }

        let shape: Vec<usize> = view.shape().to_vec();
        let tensor = DynTensor::new(&float_data, &shape, &Device::Cpu)?;
        tensors.insert(name.clone(), tensor);
    }

    Ok(VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu))
}

#[cfg(test)]
#[path = "whisper_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mel_tests.rs"]
mod mel_tests;

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod encoder_tests;

#[cfg(test)]
#[path = "decoder_architecture_tests.rs"]
mod decoder_architecture_tests;

#[cfg(test)]
#[path = "audio_mel_tests.rs"]
mod audio_mel_tests;

#[cfg(test)]
#[path = "mel_spectrogram_tests.rs"]
mod mel_spectrogram_tests;

#[cfg(test)]
#[path = "model_config_tests.rs"]
mod model_config_tests;

#[cfg(test)]
#[path = "decoding_tests.rs"]
mod decoding_tests;

#[cfg(test)]
#[path = "whisper_extended_tests.rs"]
mod whisper_extended_tests;

#[cfg(test)]
#[path = "whisper_encoder_decoder_tests.rs"]
mod whisper_encoder_decoder_tests;

#[cfg(test)]
#[path = "whisper_config_arch_extended_tests.rs"]
mod whisper_config_arch_extended_tests;

#[cfg(test)]
#[path = "whisper_pipeline_extended_tests.rs"]
mod whisper_pipeline_extended_tests;

#[cfg(test)]
#[path = "whisper_config_tests.rs"]
mod whisper_config_tests;

#[cfg(test)]
#[path = "whisper_architecture_extended_tests.rs"]
mod whisper_architecture_extended_tests;

#[cfg(test)]
#[path = "whisper_model_architecture_extended_tests.rs"]
mod whisper_model_architecture_extended_tests;

#[cfg(kani)]
#[path = "kani_whisper.rs"]
mod kani_proofs;

#[cfg(kani)]
#[path = "kani_lib_proofs.rs"]
mod kani_lib_proofs;

#[cfg(kani)]
mod kani_audio_mel_proofs;
#[cfg(kani)]
mod kani_decoder_proofs;
#[cfg(kani)]
mod kani_model_config_loading_proofs;
#[cfg(kani)]
mod kani_safetensors_proofs;
#[cfg(kani)]
mod kani_tokenizer_validation_proofs;
#[cfg(kani)]
mod kani_quality_decode_helpers_proofs;
#[cfg(kani)]
#[path = "kani_whisper_extra_proofs.rs"]
mod kani_whisper_extra_proofs;
#[cfg(kani)]
mod kani_whisper_shape_proofs;
#[cfg(kani)]
mod kani_quality_wer_proofs;
#[cfg(kani)]
mod kani_decode_long_proofs;
#[cfg(kani)]
mod kani_compression_ratio_proofs;
#[cfg(kani)]
mod kani_tokenizer_special_proofs;
#[cfg(kani)]
mod kani_beam_config_proofs;

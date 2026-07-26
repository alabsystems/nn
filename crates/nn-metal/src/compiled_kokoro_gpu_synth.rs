// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU synthesis backends for [`KokoroTextPipeline`].
//!
//! [`GpuSynth`] wraps a [`CompiledKokoro`] + [`PipelineCache`] and implements
//! the [`KokoroSynth`] trait so the backend-agnostic pipeline can dispatch
//! synthesis to the GPU.
//!
//! [`ChorusGpuSynth`] wraps a [`KokoroChorus`] + [`PipelineCache`] and
//! overrides [`KokoroSynth::synthesize_batch`] to use shared encoding —
//! encoding once per chunk, then decoding per voice (Steps 3-8 only).
//!
//! # Example
//!
//! ```rust,no_run
//! use nn_metal::compiled_kokoro::{CompiledKokoro, GpuSynth};
//! use nn_metal::PipelineCache;
//! use nn_models::kokoro_pipeline::KokoroTextPipeline;
//!
//! let mut kokoro = CompiledKokoro::new(model)?;
//! let cache = PipelineCache::new();
//! let mut gpu = GpuSynth::new(&mut kokoro, &cache);
//! // gpu implements KokoroSynth — plug into KokoroTextPipeline
//! ```

use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_core::DynTensor;
use nn_models::kokoro_error::validate_speed;
use nn_models::kokoro_pipeline::{KokoroSynth, KokoroTextPipeline, PipelineError};
use nn_models::kokoro_streaming::{AudioChunk, KokoroStreamConfig};
use nn_models::kokoro_voice_pack::VoicePack;

use super::chorus::{
    extract_pcm_from_audio, run_voice_pipeline, verify_and_extract_pcm, DefaultArenaCheckpoint,
    KokoroChorus,
};
use super::{gpu, validate_input_ids, CompiledKokoro, CompiledKokoroError};
use crate::cache::PipelineCache;

/// GPU synthesis backend wrapping [`CompiledKokoro`].
///
/// Holds mutable access to the compiled pipeline and a reference to the
/// pipeline cache. Implements [`KokoroSynth`] so it can be plugged into
/// [`KokoroTextPipeline`](nn_models::kokoro_pipeline::KokoroTextPipeline).
pub struct GpuSynth<'a> {
    kokoro: &'a mut CompiledKokoro,
    cache: &'a PipelineCache,
}

impl<'a> GpuSynth<'a> {
    /// Create a GPU synthesis backend from a compiled Kokoro pipeline.
    pub fn new(kokoro: &'a mut CompiledKokoro, cache: &'a PipelineCache) -> Self {
        Self { kokoro, cache }
    }

    /// Access the underlying compiled pipeline.
    #[must_use]
    pub fn compiled(&self) -> &CompiledKokoro {
        self.kokoro
    }

    /// Mutable access to the underlying compiled pipeline.
    pub fn compiled_mut(&mut self) -> &mut CompiledKokoro {
        self.kokoro
    }
}

impl KokoroSynth for GpuSynth<'_> {
    type Error = CompiledKokoroError;

    fn reset_sinegen_phase(&mut self) {
        self.kokoro.reset_sinegen_phase();
    }

    fn synthesize_chunk(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        let (audio, _cert) = self
            .kokoro
            .synthesize(input_ids, style, speed, self.cache)?;
        extract_pcm_from_audio(&audio)
    }
}

// -- ChorusGpuSynth: batch-aware GPU backend ----------------------------------

/// GPU chorus synthesis backend with shared-encoding optimization.
///
/// Wraps a [`KokoroChorus`] and overrides [`KokoroSynth::synthesize_batch`]
/// to encode once per chunk (Steps 1-2), then decode per voice (Steps 3-8).
/// For an 8-voice chorus, this eliminates 7 redundant encoding passes per
/// chunk (~66ms GPU saved at D=512 SEQ_LEN=32).
///
/// Use this as the synthesis backend for [`KokoroTextPipeline`] when
/// running multi-voice chorus via `text_to_chorus`.
///
/// Part of #3351 (D3), design: `designs/2026-03-23-chorus-batch-synthesis-trait.md`.
pub struct ChorusGpuSynth<'a> {
    chorus: &'a mut KokoroChorus,
    cache: &'a PipelineCache,
}

impl<'a> ChorusGpuSynth<'a> {
    /// Create a chorus GPU synthesis backend.
    pub fn new(chorus: &'a mut KokoroChorus, cache: &'a PipelineCache) -> Self {
        Self { chorus, cache }
    }

    /// Access the underlying chorus pool.
    #[must_use]
    pub fn chorus(&self) -> &KokoroChorus {
        self.chorus
    }

    /// Mutable access to the underlying chorus pool.
    pub fn chorus_mut(&mut self) -> &mut KokoroChorus {
        self.chorus
    }
}

impl KokoroSynth for ChorusGpuSynth<'_> {
    type Error = CompiledKokoroError;

    fn reset_sinegen_phase(&mut self) {
        // Reset phase on all chorus voices so each voice gets independent
        // phase continuity across chunks within a streaming session.
        for i in 0..self.chorus.n_voices() {
            if let Some(voice) = self.chorus.voice_mut(i) {
                voice.reset_sinegen_phase();
            }
        }
    }

    fn synthesize_chunk(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        // Single-voice fallback: synthesize on voice[0].
        let voice = self
            .chorus
            .voice_mut(0)
            .ok_or_else(|| CompiledKokoroError::InvalidInput("chorus has no voices".into()))?;
        let (audio, _cert) = voice.synthesize(input_ids, style, speed, self.cache)?;
        extract_pcm_from_audio(&audio)
    }

    fn synthesize_batch(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        speeds: &[f32],
    ) -> Result<Vec<Vec<f32>>, CompiledKokoroError> {
        // Shared encoding: Steps 1-2 once on voice[0], Steps 3-8 per voice.
        // Uses public KokoroChorus + CompiledKokoro API.
        let n = self.chorus.n_voices();
        if styles.len() != n {
            return Err(CompiledKokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            )));
        }
        if speeds.len() != n {
            return Err(CompiledKokoroError::InvalidInput(format!(
                "speeds length {} != n_voices {n}",
                speeds.len()
            )));
        }
        for &s in speeds {
            validate_speed(s)?;
        }
        crate::arena::pool_reclaim();

        // Pre-split styles and upload to GPU.
        let v0 = self
            .chorus
            .voice(0)
            .ok_or_else(|| CompiledKokoroError::InvalidInput("chorus has no voices".into()))?;
        validate_input_ids(input_ids, v0.config().plbert.max_position_embeddings)?;

        // Pre-size the arena for the shared-encode decode loop (#4289).
        // Multiply by 2 to cover shared Phase 1/regulate state plus one
        // per-voice decode window. The per-voice checkpoint guard below
        // reclaims each decode window before the next voice runs.
        let arena_estimate = v0.estimate_arena_bytes().saturating_mul(2);
        if arena_estimate > 0 {
            let _ =
                crate::arena::ensure_default_arena_capacity(self.cache.context(), arena_estimate);
        }

        let mut decoder_styles: Vec<DynTensor> = Vec::with_capacity(n);
        let mut prosody_styles: Vec<DynTensor> = Vec::with_capacity(n);
        for style in styles {
            let split = v0.split_style(style)?;
            decoder_styles.push(split.decoder_style.to_device(&gpu())?);
            prosody_styles.push(split.prosody_style.to_device(&gpu())?);
        }

        let result = (|| -> Result<Vec<Vec<f32>>, CompiledKokoroError> {
            let audio_tensors: Vec<DynTensor> = with_nan_check_policy(
                NanCheckPolicy::Skip,
                || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                    // Steps 1-2: encode once on voice[0].
                    let v0 = self.chorus.voice_mut(0).ok_or_else(|| {
                        CompiledKokoroError::InvalidInput("chorus has no voices".into())
                    })?;
                    let enc = v0.step_encode(input_ids, self.cache)?;

                    // Steps 3-8 per voice using shared encoding.
                    let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                    for i in 0..n {
                        let voice = self.chorus.voice_mut(i).ok_or_else(|| {
                            CompiledKokoroError::InvalidInput(format!(
                                "chorus voice {i} not available"
                            ))
                        })?;
                        let _arena_cp = DefaultArenaCheckpoint::new();
                        let audio = run_voice_pipeline(
                            voice,
                            &enc,
                            &prosody_styles[i],
                            &decoder_styles[i],
                            speeds[i],
                            self.cache,
                        )?;
                        audios.push(audio);
                    }
                    Ok(audios)
                },
            )?;

            // NaN guard + verify + extract PCM per voice.
            verify_and_extract_pcm(&self.chorus.voices, &audio_tensors)
        })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }
}

// -- Convenience methods on CompiledKokoro ------------------------------------

impl CompiledKokoro {
    /// Text to audio via GPU: constructs a temporary pipeline internally.
    ///
    /// This is a convenience wrapper around [`KokoroTextPipeline`] + [`GpuSynth`].
    /// For repeated calls, prefer constructing a pipeline once and reusing it.
    ///
    /// `phonemize` is a closure because `EspeakEngine` is `!Send` — the caller
    /// controls espeak FFI lifetime. See [`KokoroTextPipeline::text_to_audio`]
    /// for patterns (espeak FFI, pre-computed phonemes, lexicon fallback).
    pub fn text_to_audio(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let gpu = GpuSynth::new(self, cache);
        let mut pipeline = KokoroTextPipeline::english(gpu);
        pipeline.text_to_audio(text, phonemize, style, speed, stream_config)
    }

    /// Text to streaming audio via GPU: delivers each chunk via callback.
    ///
    /// Like [`text_to_audio`](Self::text_to_audio) but calls `on_chunk` after
    /// each chunk is synthesized and crossfaded. Enables sub-utterance playback
    /// latency — the first chunk is delivered after 1 synthesis pass, not after
    /// the full utterance completes.
    pub fn text_to_audio_streaming(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        style: &DynTensor,
        speed: f32,
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
        on_chunk: impl FnMut(&AudioChunk),
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let gpu = GpuSynth::new(self, cache);
        let mut pipeline = KokoroTextPipeline::english(gpu);
        pipeline.text_to_audio_streaming(text, phonemize, style, speed, stream_config, on_chunk)
    }

    /// Text to audio using a named voice from a [`VoicePack`].
    ///
    /// Looks up `voice_name` in the pack, then delegates to
    /// [`text_to_audio`](Self::text_to_audio).
    pub fn text_to_audio_named(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        voice_name: &str,
        voice_pack: &VoicePack,
        speed: f32,
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let style = voice_pack
            .get_or_err(voice_name)
            .map_err(PipelineError::Assembly)?
            .clone();
        self.text_to_audio(text, phonemize, &style, speed, cache, stream_config)
    }

    /// Text to streaming audio using a named voice from a [`VoicePack`].
    ///
    /// Looks up `voice_name` in the pack, then delegates to
    /// [`text_to_audio_streaming`](Self::text_to_audio_streaming).
    pub fn text_to_audio_streaming_named(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        voice_name: &str,
        voice_pack: &VoicePack,
        speed: f32,
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
        on_chunk: impl FnMut(&AudioChunk),
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let style = voice_pack
            .get_or_err(voice_name)
            .map_err(PipelineError::Assembly)?
            .clone();
        self.text_to_audio_streaming(
            text,
            phonemize,
            &style,
            speed,
            cache,
            stream_config,
            on_chunk,
        )
    }
}

// -- Convenience methods on KokoroChorus --------------------------------------

impl KokoroChorus {
    /// Text to multi-voice chorus via GPU.
    ///
    /// Convenience wrapper: creates a [`ChorusGpuSynth`] + [`KokoroTextPipeline`],
    /// then delegates to [`KokoroTextPipeline::text_to_chorus`]. For repeated
    /// calls, prefer constructing a pipeline once with [`ChorusGpuSynth`].
    pub fn text_to_chorus(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        styles: &[DynTensor],
        speeds: &[f32],
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let config = self.config.clone();
        let gpu = ChorusGpuSynth::new(self, cache);
        let mut pipeline = KokoroTextPipeline::english(gpu);
        pipeline.text_to_chorus(text, phonemize, styles, speeds, &config, stream_config)
    }

    /// Text to streaming multi-voice chorus via GPU with per-chunk callback.
    ///
    /// Like [`text_to_chorus`](Self::text_to_chorus) but calls `on_chunk` after
    /// each chunk's voices are mixed and crossfaded. Enables sub-utterance
    /// playback latency for multi-voice chorus.
    ///
    /// First-audio latency = 1 chunk × N voices (via shared-encode batch),
    /// not full utterance × N voices.
    pub fn text_to_chorus_streaming(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        styles: &[DynTensor],
        speeds: &[f32],
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
        on_chunk: impl FnMut(&AudioChunk),
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let config = self.config.clone();
        let gpu = ChorusGpuSynth::new(self, cache);
        let mut pipeline = KokoroTextPipeline::english(gpu);
        pipeline.text_to_chorus_streaming(
            text,
            phonemize,
            styles,
            speeds,
            &config,
            stream_config,
            on_chunk,
        )
    }

    /// Text to chorus using named voices from a [`VoicePack`].
    ///
    /// Looks up each voice name, then delegates to
    /// [`text_to_chorus`](Self::text_to_chorus).
    pub fn text_to_chorus_named(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        voice_names: &[&str],
        voice_pack: &VoicePack,
        speeds: &[f32],
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let styles: Vec<DynTensor> = voice_names
            .iter()
            .map(|name| {
                voice_pack
                    .get_or_err(name)
                    .cloned()
                    .map_err(PipelineError::Assembly)
            })
            .collect::<Result<_, _>>()?;
        self.text_to_chorus(text, phonemize, &styles, speeds, cache, stream_config)
    }

    /// Text to streaming chorus using named voices from a [`VoicePack`].
    ///
    /// Looks up each voice name, then delegates to
    /// [`text_to_chorus_streaming`](Self::text_to_chorus_streaming).
    pub fn text_to_chorus_streaming_named(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        voice_names: &[&str],
        voice_pack: &VoicePack,
        speeds: &[f32],
        cache: &PipelineCache,
        stream_config: &KokoroStreamConfig,
        on_chunk: impl FnMut(&AudioChunk),
    ) -> Result<Vec<AudioChunk>, PipelineError<CompiledKokoroError>> {
        let styles: Vec<DynTensor> = voice_names
            .iter()
            .map(|name| {
                voice_pack
                    .get_or_err(name)
                    .cloned()
                    .map_err(PipelineError::Assembly)
            })
            .collect::<Result<_, _>>()?;
        self.text_to_chorus_streaming(
            text,
            phonemize,
            &styles,
            speeds,
            cache,
            stream_config,
            on_chunk,
        )
    }
}

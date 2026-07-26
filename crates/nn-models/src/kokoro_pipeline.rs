// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic text-to-audio pipeline for Kokoro TTS.
//!
//! Composes: TextPreprocessor → G2P → Remapper → Tokenizer → Synthesizer → Assembly.
//!
//! The pipeline is generic over a `KokoroSynth` backend so that both CPU
//! (`KokoroModel`) and GPU (`CompiledKokoro` via nn-metal) can be plugged in.

use nn_core::{Device, DynTensor};

use crate::kokoro_chorus::{mix_voices_with_config, ChorusConfig};
use crate::kokoro_error::KokoroError;
use crate::kokoro_g2p::EspeakRemapper;
use crate::kokoro_streaming::{
    assemble_streaming_chorus, assemble_streaming_chunks, AudioChunk, KokoroStreamConfig,
    StreamingAssembler, StreamingKokoroSession,
};
use crate::kokoro_text_preprocess::TextPreprocessor;
use crate::kokoro_tokenizer::KokoroTokenizer;
use crate::kokoro_tts::EncoderFeaturesResult;
use crate::kokoro_voice_pack::VoicePack;

/// Synthesis backend trait — CPU or GPU.
///
/// Implementors convert a single chunk of token IDs to PCM audio.
pub trait KokoroSynth {
    type Error: std::error::Error + Send + 'static;

    /// Synthesize a single chunk: token IDs → PCM audio.
    ///
    /// `input_ids` is `[1, T]` (batch=1, seq_len=T) u32 tensor.
    /// `style` is `[1, 2*style_dim]` style embedding.
    fn synthesize_chunk(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> Result<Vec<f32>, Self::Error>;

    /// Reset the SineGen cumulative phase for a new streaming session.
    ///
    /// Call before synthesizing the first chunk of a streaming session.
    /// Ensures the first chunk starts with zero phase and subsequent chunks
    /// carry phase continuity from SineGen's cumulative sum.
    ///
    /// Default: no-op (for backends without stateful SineGen phase).
    fn reset_sinegen_phase(&mut self) {}

    /// Synthesize one chunk for multiple voices. Default: sequential.
    ///
    /// Backends with multi-voice support (e.g., GPU with shared encoding)
    /// override this to share encoding work across voices.
    ///
    /// `styles` and `speeds` must have equal lengths. The number of returned
    /// PCM buffers equals `styles.len()`. Callers (e.g., `text_to_chorus`)
    /// are responsible for ensuring this matches the expected voice count.
    ///
    /// Returns one PCM buffer per voice.
    fn synthesize_batch(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        speeds: &[f32],
    ) -> Result<Vec<Vec<f32>>, Self::Error> {
        styles
            .iter()
            .zip(speeds)
            .map(|(s, &sp)| self.synthesize_chunk(input_ids, s, sp))
            .collect()
    }

    /// Extract frozen encoder features without running the decoder/vocoder.
    ///
    /// Runs the encoder portion only: PlBert → bert_encoder → TextEncoder →
    /// ProsodyPredictor → length_regulate. Returns the regulated encoder
    /// features `[B, d_en, T_mel]` needed for singing decoder LoRA training.
    ///
    /// # Arguments
    ///
    /// * `input_ids` — `[1, T]` phoneme token IDs.
    /// * `style` — `[1, 2*style_dim]` voice embedding.
    /// * `speed` — Speaking rate multiplier (1.0 = normal).
    ///
    /// # Returns
    ///
    /// [`EncoderFeaturesResult`] with regulated features, aligned prosody
    /// features, and predicted durations.
    ///
    /// Default: returns `None` (not all backends support encoder-only forward).
    fn extract_encoder_features(
        &self,
        _input_ids: &DynTensor,
        _style: &DynTensor,
        _speed: f32,
    ) -> Result<Option<EncoderFeaturesResult>, Self::Error> {
        Ok(None)
    }
}

/// CPU synthesis backend via `KokoroModel::forward_audio`.
impl KokoroSynth for crate::kokoro_tts::KokoroModel {
    type Error = KokoroError;

    fn reset_sinegen_phase(&mut self) {
        // Call the inherent method on KokoroModel, not the trait method.
        Self::reset_sinegen_phase(self);
    }

    fn synthesize_chunk(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> Result<Vec<f32>, KokoroError> {
        let audio = self.forward_audio(input_ids, style, speed)?;
        audio.to_flat_vec::<f32>().map_err(KokoroError::Tensor)
    }

    fn extract_encoder_features(
        &self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speed: f32,
    ) -> Result<Option<EncoderFeaturesResult>, KokoroError> {
        Ok(Some(
            self.forward_encoder_features(input_ids, style, speed)?,
        ))
    }
}

/// Typed error for the text-to-audio pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PipelineError<S: std::error::Error + 'static> {
    #[error("phonemization failed: {0}")]
    Phonemization(Box<dyn std::error::Error + Send + Sync>),
    #[error("tokenization produced no chunks from input text")]
    EmptyChunks,
    #[error("tensor construction failed: {0}")]
    Tensor(#[from] nn_core::TensorError),
    #[error("synthesis failed: {0}")]
    Synthesis(S),
    #[error("audio assembly failed: {0}")]
    Assembly(KokoroError),
}

/// Backend-agnostic text-to-audio pipeline for Kokoro TTS.
///
/// Composes: TextPreprocessor → phonemize (caller-provided) → Remapper → Tokenizer → Synth → Assembly.
///
/// The `phonemize` closure is caller-provided (not owned) because `EspeakEngine`
/// is `!Send + !Sync` (unsafe FFI + global C state). This keeps the pipeline
/// `Send`-able when the synth backend is `Send`.
pub struct KokoroTextPipeline<S: KokoroSynth> {
    preprocessor: TextPreprocessor,
    remapper: EspeakRemapper,
    tokenizer: KokoroTokenizer,
    synth: S,
}

impl<S: KokoroSynth> KokoroTextPipeline<S> {
    /// Create a new pipeline with the given components.
    pub fn new(
        preprocessor: TextPreprocessor,
        remapper: EspeakRemapper,
        tokenizer: KokoroTokenizer,
        synth: S,
    ) -> Self {
        Self {
            preprocessor,
            remapper,
            tokenizer,
            synth,
        }
    }

    /// Create a pipeline with default English settings.
    pub fn english(synth: S) -> Self {
        Self {
            preprocessor: TextPreprocessor::english(),
            remapper: EspeakRemapper::english_us(),
            tokenizer: KokoroTokenizer::kokoro_default(),
            synth,
        }
    }

    /// Access the synthesizer backend.
    #[must_use]
    pub fn synth(&self) -> &S {
        &self.synth
    }

    /// Mutable access to the synthesizer backend.
    pub fn synth_mut(&mut self) -> &mut S {
        &mut self.synth
    }

    /// Access the tokenizer.
    #[must_use]
    pub fn tokenizer(&self) -> &KokoroTokenizer {
        &self.tokenizer
    }

    /// Access the preprocessor.
    #[must_use]
    pub fn preprocessor(&self) -> &TextPreprocessor {
        &self.preprocessor
    }

    /// Access the remapper.
    #[must_use]
    pub fn remapper(&self) -> &EspeakRemapper {
        &self.remapper
    }

    /// Full text → audio: preprocess, phonemize, tokenize, chunk, synthesize, crossfade.
    ///
    /// `phonemize` is a closure so the caller controls espeak FFI or provides
    /// pre-computed phonemes. This avoids the pipeline owning espeak state.
    ///
    /// # Patterns
    ///
    /// ```ignore
    /// // Pattern A: espeak FFI (feature-gated)
    /// let engine = EspeakEngine::new("en-us")?;
    /// pipeline.text_to_audio(text, |t| Ok(engine.text_to_ipa(t)?), style, speed, &config)?;
    ///
    /// // Pattern B: pre-computed phonemes
    /// pipeline.text_to_audio(text, |_| Ok("hɛˈloʊ wˈɜːɹld".into()), style, speed, &config)?;
    /// ```
    pub fn text_to_audio(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        style: &DynTensor,
        speed: f32,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<S::Error>> {
        // 1. Preprocess text (normalize punctuation, expand abbreviations/numbers)
        let cleaned = self.preprocessor.preprocess(text);

        // 2. Phonemize (espeak FFI or pre-computed)
        let ipa = phonemize(&cleaned).map_err(PipelineError::Phonemization)?;

        // 3. Remap IPA → Kokoro phonemes
        let phonemes = self.remapper.remap(&ipa);

        // 4. Chunk and tokenize (waterfall split at punctuation, respects 510 token limit)
        let chunks = self.tokenizer.chunk_and_encode(&phonemes);
        if chunks.is_empty() {
            return Err(PipelineError::EmptyChunks);
        }

        // 5. Convert token chunks to DynTensors
        let input_tensors = chunks_to_tensors(&chunks)?;

        // 5b. Reset SineGen phase for this new utterance. Subsequent
        // synthesize_chunk calls carry phase continuity automatically.
        self.synth.reset_sinegen_phase();

        // 6. Synthesize each chunk
        let mut raw_pcm: Vec<Vec<f32>> = Vec::with_capacity(input_tensors.len());
        for input_ids in &input_tensors {
            raw_pcm.push(
                self.synth
                    .synthesize_chunk(input_ids, style, speed)
                    .map_err(PipelineError::Synthesis)?,
            );
        }

        // 7. Assemble with crossfade
        assemble_streaming_chunks(&raw_pcm, stream_config).map_err(PipelineError::Assembly)
    }

    /// Streaming text → audio: calls `on_chunk` after each synthesized chunk.
    ///
    /// Unlike [`text_to_audio`](Self::text_to_audio) which returns all chunks at
    /// once after full synthesis, this method calls `on_chunk` after each chunk is
    /// synthesized and assembled via [`StreamingAssembler`]. This enables
    /// sub-utterance playback latency: start playing the first chunk while
    /// subsequent chunks are still being synthesized.
    ///
    /// The callback receives each [`AudioChunk`] immediately. The full
    /// `Vec<AudioChunk>` is also returned for total-audio metadata.
    ///
    /// # Arguments
    ///
    /// * `on_chunk` - Called with each assembled `AudioChunk` as it becomes
    ///   available. The callback is infallible — use it for playback, logging,
    ///   or buffering. For fallible consumers, buffer internally and check after.
    pub fn text_to_audio_streaming(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        style: &DynTensor,
        speed: f32,
        stream_config: &KokoroStreamConfig,
        mut on_chunk: impl FnMut(&AudioChunk),
    ) -> Result<Vec<AudioChunk>, PipelineError<S::Error>> {
        let cleaned = self.preprocessor.preprocess(text);
        let ipa = phonemize(&cleaned).map_err(PipelineError::Phonemization)?;
        let phonemes = self.remapper.remap(&ipa);
        let chunks = self.tokenizer.chunk_and_encode(&phonemes);
        if chunks.is_empty() {
            return Err(PipelineError::EmptyChunks);
        }
        let input_tensors = chunks_to_tensors(&chunks)?;
        let n = input_tensors.len();

        // Reset SineGen phase for this new streaming utterance.
        self.synth.reset_sinegen_phase();

        let mut assembler =
            StreamingAssembler::new(stream_config.clone(), n).map_err(PipelineError::Assembly)?;
        let mut result = Vec::with_capacity(n);

        for input_ids in &input_tensors {
            let raw_pcm = self
                .synth
                .synthesize_chunk(input_ids, style, speed)
                .map_err(PipelineError::Synthesis)?;
            let audio_chunk = assembler
                .push_raw(raw_pcm)
                .map_err(PipelineError::Assembly)?;
            on_chunk(&audio_chunk);
            result.push(audio_chunk);
        }

        Ok(result)
    }

    /// Create a pull-based streaming session from text.
    ///
    /// Pre-processes, phonemizes, tokenizes, and chunks the text, then wraps
    /// the result in a [`StreamingKokoroSession`] ready for pull-based
    /// iteration via [`next_chunk()`](StreamingKokoroSession::next_chunk).
    ///
    /// This is the main entry point for dvoice's conductor, which needs
    /// caller-controlled iteration rather than the callback-driven
    /// [`text_to_audio_streaming()`](Self::text_to_audio_streaming).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let session = pipeline.create_streaming_session(text, phonemize, &config)?;
    /// while let Some(chunk) = session.next_chunk(&mut synth, &style, speed)? {
    ///     play(chunk.pcm());
    /// }
    /// ```
    pub fn create_streaming_session(
        &self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        stream_config: &KokoroStreamConfig,
    ) -> Result<StreamingKokoroSession, PipelineError<S::Error>> {
        let cleaned = self.preprocessor.preprocess(text);
        let ipa = phonemize(&cleaned).map_err(PipelineError::Phonemization)?;
        let phonemes = self.remapper.remap(&ipa);
        let chunks = self.tokenizer.chunk_and_encode(&phonemes);
        if chunks.is_empty() {
            return Err(PipelineError::EmptyChunks);
        }
        let tensors = chunks_to_tensors(&chunks)?;
        StreamingKokoroSession::new(tensors, stream_config.clone()).map_err(PipelineError::Assembly)
    }

    /// Text → phoneme token chunks (for callers that want to control synthesis).
    ///
    /// Returns `Vec<(phoneme_text, token_ids)>` where each entry fits within
    /// the 510-token limit.
    pub fn text_to_tokens(
        &self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
    ) -> Result<Vec<(String, Vec<u32>)>, PipelineError<S::Error>> {
        let cleaned = self.preprocessor.preprocess(text);
        let ipa = phonemize(&cleaned).map_err(PipelineError::Phonemization)?;
        let phonemes = self.remapper.remap(&ipa);
        let chunks = self.tokenizer.chunk_and_encode(&phonemes);
        if chunks.is_empty() {
            return Err(PipelineError::EmptyChunks);
        }
        Ok(chunks)
    }

    /// Multi-voice text → mixed audio with streaming crossfade.
    ///
    /// Text processing (preprocess → phonemize → tokenize) is shared across
    /// all voices. Each voice synthesizes the same token chunks with its own
    /// style embedding and speed, then outputs are mixed per-chunk and
    /// crossfaded at chunk boundaries.
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to synthesize.
    /// * `phonemize` - Caller-provided phonemization closure (see [`text_to_audio`](Self::text_to_audio)).
    /// * `styles` - Per-voice style embeddings. Length must equal `chorus_config.n_voices`.
    /// * `speeds` - Per-voice speed multipliers. Length must equal `chorus_config.n_voices`.
    /// * `chorus_config` - Voice count, per-voice gains, clipping.
    /// * `stream_config` - Crossfade configuration.
    pub fn text_to_chorus(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        styles: &[DynTensor],
        speeds: &[f32],
        chorus_config: &ChorusConfig,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<S::Error>> {
        chorus_config.validate().map_err(PipelineError::Assembly)?;
        let n = chorus_config.n_voices;
        if styles.len() != n {
            return Err(PipelineError::Assembly(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))));
        }
        if speeds.len() != n {
            return Err(PipelineError::Assembly(KokoroError::InvalidInput(format!(
                "speeds length {} != n_voices {n}",
                speeds.len()
            ))));
        }

        // 1. Shared text processing — all voices get the same token chunks.
        let cleaned = self.preprocessor.preprocess(text);
        let ipa = phonemize(&cleaned).map_err(PipelineError::Phonemization)?;
        let phonemes = self.remapper.remap(&ipa);
        let chunks = self.tokenizer.chunk_and_encode(&phonemes);
        if chunks.is_empty() {
            return Err(PipelineError::EmptyChunks);
        }
        let input_tensors = chunks_to_tensors(&chunks)?;

        // Reset SineGen phase for this new chorus utterance.
        self.synth.reset_sinegen_phase();

        // 2. Synthesize chunk-major: all voices per chunk via synthesize_batch.
        //    Chunk-major order enables shared-encoding GPU backends and produces
        //    the first mixable output sooner (after N voices, not N×C calls).
        let n_chunks = input_tensors.len();
        let mut per_voice_chunks: Vec<Vec<Vec<f32>>> =
            (0..n).map(|_| Vec::with_capacity(n_chunks)).collect();

        for input_ids in &input_tensors {
            let batch_pcm = self
                .synth
                .synthesize_batch(input_ids, styles, speeds)
                .map_err(PipelineError::Synthesis)?;

            for (vi, pcm) in batch_pcm.into_iter().enumerate() {
                per_voice_chunks[vi].push(pcm);
            }
        }

        // 3. Mix per-chunk across voices and apply crossfade.
        assemble_streaming_chorus(&per_voice_chunks, chorus_config, stream_config)
            .map_err(PipelineError::Assembly)
    }

    /// Multi-voice text → mixed audio with per-chunk callback delivery.
    ///
    /// Like [`text_to_chorus`](Self::text_to_chorus) but calls `on_chunk` after
    /// each chunk's voices are mixed and crossfaded. Enables sub-utterance
    /// playback latency for multi-voice chorus.
    ///
    /// First-audio latency = 1 chunk × N voices (via `synthesize_batch`),
    /// not C chunks × N voices.
    pub fn text_to_chorus_streaming(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        styles: &[DynTensor],
        speeds: &[f32],
        chorus_config: &ChorusConfig,
        stream_config: &KokoroStreamConfig,
        mut on_chunk: impl FnMut(&AudioChunk),
    ) -> Result<Vec<AudioChunk>, PipelineError<S::Error>> {
        chorus_config.validate().map_err(PipelineError::Assembly)?;
        let n = chorus_config.n_voices;
        if styles.len() != n {
            return Err(PipelineError::Assembly(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))));
        }
        if speeds.len() != n {
            return Err(PipelineError::Assembly(KokoroError::InvalidInput(format!(
                "speeds length {} != n_voices {n}",
                speeds.len()
            ))));
        }

        // Shared text processing.
        let cleaned = self.preprocessor.preprocess(text);
        let ipa = phonemize(&cleaned).map_err(PipelineError::Phonemization)?;
        let phonemes = self.remapper.remap(&ipa);
        let chunks = self.tokenizer.chunk_and_encode(&phonemes);
        if chunks.is_empty() {
            return Err(PipelineError::EmptyChunks);
        }
        let input_tensors = chunks_to_tensors(&chunks)?;
        let n_chunks = input_tensors.len();

        // Reset SineGen phase for this new chorus streaming utterance.
        self.synth.reset_sinegen_phase();

        // For stereo chorus, double crossfade_samples to cover the same time
        // duration in the interleaved float buffer — matches assemble_streaming_chorus.
        let is_stereo = chorus_config.pans.is_some();
        let channels: usize = if is_stereo { 2 } else { 1 };
        let effective_config = if is_stereo {
            KokoroStreamConfig {
                crossfade_samples: stream_config.crossfade_samples * 2,
                crossfade_window: stream_config.crossfade_window,
            }
        } else {
            stream_config.clone()
        };

        let mut assembler =
            StreamingAssembler::new(effective_config, n_chunks).map_err(PipelineError::Assembly)?;
        let mut result = Vec::with_capacity(n_chunks);

        for input_ids in &input_tensors {
            // Synthesize all voices for this chunk.
            let batch_pcm = self
                .synth
                .synthesize_batch(input_ids, styles, speeds)
                .map_err(PipelineError::Synthesis)?;

            // Mix per-voice PCM into a single mixed buffer.
            let mixed = mix_voices_with_config(&batch_pcm, chorus_config)
                .map_err(PipelineError::Assembly)?;

            // Crossfade with previous chunk and emit.
            let mut audio_chunk = assembler.push_raw(mixed).map_err(PipelineError::Assembly)?;
            audio_chunk.channels = channels;
            on_chunk(&audio_chunk);
            result.push(audio_chunk);
        }

        Ok(result)
    }

    /// Convenience: text → audio using a named voice from a [`VoicePack`].
    ///
    /// Looks up `voice_name` in `voice_pack`, then delegates to
    /// [`text_to_audio`](Self::text_to_audio).
    pub fn text_to_audio_named(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        voice_name: &str,
        voice_pack: &VoicePack,
        speed: f32,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<S::Error>> {
        let style = voice_pack
            .get_or_err(voice_name)
            .map_err(PipelineError::Assembly)?;
        self.text_to_audio(text, phonemize, style, speed, stream_config)
    }

    /// Convenience: multi-voice chorus using named voices from a [`VoicePack`].
    ///
    /// Looks up each voice name in `voice_pack`, then delegates to
    /// [`text_to_chorus`](Self::text_to_chorus).
    pub fn text_to_chorus_named(
        &mut self,
        text: &str,
        phonemize: impl Fn(&str) -> Result<String, Box<dyn std::error::Error + Send + Sync>>,
        voice_names: &[&str],
        voice_pack: &VoicePack,
        speeds: &[f32],
        chorus_config: &ChorusConfig,
        stream_config: &KokoroStreamConfig,
    ) -> Result<Vec<AudioChunk>, PipelineError<S::Error>> {
        let styles: Vec<DynTensor> = voice_names
            .iter()
            .map(|name| {
                voice_pack
                    .get_or_err(name)
                    .cloned()
                    .map_err(PipelineError::Assembly)
            })
            .collect::<Result<_, _>>()?;
        self.text_to_chorus(
            text,
            phonemize,
            &styles,
            speeds,
            chorus_config,
            stream_config,
        )
    }
}

/// Convert tokenizer output to DynTensors suitable for synthesis.
///
/// Each `(phoneme_text, token_ids)` becomes a `[1, T]` u32 tensor on CPU.
pub fn chunks_to_tensors(
    chunks: &[(String, Vec<u32>)],
) -> Result<Vec<DynTensor>, nn_core::TensorError> {
    chunks
        .iter()
        .map(|(_, ids)| {
            let len = ids.len();
            DynTensor::from_vec_u32(ids.clone(), &[1, len], &Device::Cpu)
        })
        .collect()
}

#[cfg(test)]
#[path = "kokoro_pipeline_tests.rs"]
mod tests;

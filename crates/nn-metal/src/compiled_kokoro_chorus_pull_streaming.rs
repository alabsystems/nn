// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pull-based streaming chorus synthesis session for [`KokoroChorus`].
//!
//! Unlike the push-based [`synthesize_streaming_chorus`](super::chorus::KokoroChorus::synthesize_streaming_chorus)
//! which synthesizes all chunks at once and returns a `Vec<AudioChunk>`,
//! `StreamingChorusSession` is a state machine that synthesizes one chunk per
//! `next_chunk()` call, mixing all voices and applying crossfade incrementally.
//!
//! The caller controls pacing, which is ideal for dvoice's chorus conductor
//! where mixed audio chunks must be delivered to the audio output on demand.
//!
//! # Design
//!
//! `CompiledKokoro` (and thus `KokoroChorus`) is `!Send`, so we cannot move
//! them to a background thread. The session owns only pre-tokenized chunks,
//! per-voice styles, and crossfade state. The caller passes `&mut KokoroChorus`
//! and `&PipelineCache` to each `next_chunk()` call.
//!
//! Crossfade is applied incrementally: the session retains the "tail" of the
//! previous mixed chunk and blends it with the head of the next mixed chunk.
//!
//! # Usage
//!
//! ```text
//! let chunks: Vec<DynTensor> = tokenizer.chunk_and_encode(text);
//! let styles: Vec<DynTensor> = voice_pack.styles();
//! let mut session = StreamingChorusSession::new(
//!     chunks, styles, 1.0, stream_config,
//! )?;
//!
//! while let Some(result) = session.next_chunk(&mut chorus, &cache) {
//!     let audio_chunk = result?;
//!     // Play or buffer the mixed audio chunk immediately.
//! }
//! ```
//!
//! Part of #4105.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_models::kokoro_chorus_dynamics::{BusLimiter, DynamicsPreset, MultibandCompressor};
use nn_models::kokoro_chorus_reverb::ReverbConfig;
use nn_models::kokoro_chorus_reverb_streaming::StreamingReverb;
use nn_models::kokoro_streaming::{AudioChunk, KokoroStreamConfig, StreamingAssembler};

use super::chorus::{
    run_voice_pipeline, verify_and_extract_pcm, DefaultArenaCheckpoint, KokoroChorus,
};
use super::precompile::PrecompileShapes;
use super::{gpu, validate_input_ids, CompiledKokoroError};
use crate::cache::PipelineCache;

/// How chunk token IDs are organized for the chorus session.
///
/// - [`SharedText`](Self::SharedText): all voices synthesize the same chunks
///   (shared encoding, Steps 1-2 once, Steps 3-8 per voice).
/// - [`PerVoice`](Self::PerVoice): each voice has its own chunk list and runs
///   the full pipeline independently.
#[derive(Debug)]
pub enum ChorusChunkMode {
    /// All voices share the same chunks (same text, different styles).
    SharedText(Vec<DynTensor>),
    /// Per-voice chunk lists. `per_voice[v][i]` is voice `v`'s chunk `i`.
    /// All inner `Vec`s must have the same length.
    PerVoice(Vec<Vec<DynTensor>>),
}

impl ChorusChunkMode {
    /// Number of chunks in the session (all voices have the same count).
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::SharedText(chunks) => chunks.len(),
            Self::PerVoice(per_voice) => per_voice.first().map_or(0, Vec::len),
        }
    }

    /// Whether there are no chunks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Collect the token sequence lengths from all chunks across all voices.
    ///
    /// For `SharedText`, returns the second dimension of each chunk tensor.
    /// For `PerVoice`, returns the second dimension of every chunk across
    /// all voices (flattened, including duplicates).
    pub(super) fn token_lengths(&self) -> Vec<usize> {
        match self {
            Self::SharedText(chunks) => chunks
                .iter()
                .filter_map(|t| {
                    let dims = t.dims();
                    if dims.len() >= 2 {
                        Some(dims[1])
                    } else {
                        None
                    }
                })
                .collect(),
            Self::PerVoice(per_voice) => per_voice
                .iter()
                .flat_map(|voice_chunks| {
                    voice_chunks.iter().filter_map(|t| {
                        let dims = t.dims();
                        if dims.len() >= 2 {
                            Some(dims[1])
                        } else {
                            None
                        }
                    })
                })
                .collect(),
        }
    }

    /// Collect token lengths only for chunks at or after `start_idx`.
    fn remaining_token_lengths(&self, start_idx: usize) -> Vec<usize> {
        if start_idx == 0 {
            return self.token_lengths();
        }
        match self {
            Self::SharedText(chunks) => chunks
                .iter()
                .skip(start_idx)
                .filter_map(|t| {
                    let dims = t.dims();
                    if dims.len() >= 2 {
                        Some(dims[1])
                    } else {
                        None
                    }
                })
                .collect(),
            Self::PerVoice(per_voice) => per_voice
                .iter()
                .flat_map(|voice_chunks| {
                    voice_chunks.iter().skip(start_idx).filter_map(|t| {
                        let dims = t.dims();
                        if dims.len() >= 2 {
                            Some(dims[1])
                        } else {
                            None
                        }
                    })
                })
                .collect(),
        }
    }

    /// Build the warmup shape plan for the remaining unsynthesized chunk
    /// range, if any valid sequence lengths are present.
    fn remaining_precompile_shapes(&self, start_idx: usize) -> Option<PrecompileShapes> {
        PrecompileShapes::from_token_lengths(&self.remaining_token_lengths(start_idx))
    }
}

/// Pull-based streaming chorus synthesis session.
///
/// Synthesizes one chunk at a time across all voices, mixes them, applies
/// crossfade with the previous chunk's tail, and returns a ready-to-play
/// [`AudioChunk`].
///
/// # State machine
///
/// The session tracks a cursor into the chunk list. Each `next_chunk()` call:
/// 1. Synthesizes the current chunk across all voices.
/// 2. Mixes per-voice audio using the chorus config's gains/pans.
/// 3. Applies crossfade with the previous chunk's tail.
/// 4. Advances the cursor and returns the mixed `AudioChunk`.
///
/// When the cursor reaches the end, `next_chunk()` returns `None`.
///
/// # Chunk modes
///
/// - **Shared text** (via [`new()`](Self::new)): all voices synthesize the same
///   chunks. Encoding is shared (Steps 1-2 once, Steps 3-8 per voice).
/// - **Per-voice text** (via [`new_varied_text()`](Self::new_varied_text)): each
///   voice has its own chunk list and runs the full pipeline independently.
pub struct StreamingChorusSession {
    /// Chunk token IDs, organized by mode.
    chunk_mode: ChorusChunkMode,
    /// Per-voice style embeddings.
    styles: Vec<DynTensor>,
    /// Speaking rate multiplier.
    speed: f32,
    /// Crossfade configuration (retained for lazy assembler init and reset).
    stream_config: KokoroStreamConfig,
    /// Current chunk cursor. When `cursor >= chunk_mode.len()`, the session is done.
    cursor: usize,
    /// Crossfade assembler. Lazily initialized on the first `next_chunk()` call
    /// because the channel count depends on the `KokoroChorus` config (mono vs
    /// stereo), which is only available at synthesis time.
    assembler: Option<StreamingAssembler>,
    /// Whether the session has been cancelled via [`cancel()`](Self::cancel).
    cancelled: bool,
    /// Whether to precompile pipeline segments for the chunk shapes before
    /// the first synthesis. Defaults to `true`. Set to `false` via
    /// [`with_precompile`](Self::with_precompile) to skip warmup for
    /// latency-sensitive cold starts.
    precompile: bool,
    /// Tracks whether the session's one-time warmup step has already been
    /// consumed, either by explicit precompile or by the automatic
    /// first-chunk path.
    precompiled: bool,
    /// Optional reverb config. When set, a [`StreamingReverb`] is lazily
    /// created on the first `next_chunk()` call (once the channel count
    /// is known) and persisted across chunks so the reverb tail carries
    /// from one chunk to the next.
    reverb_config: Option<ReverbConfig>,
    /// Persistent reverb processor. Lazily initialized alongside the
    /// assembler on the first `next_chunk()` call.
    streaming_reverb: Option<StreamingReverb>,
    /// Optional dynamics processing preset. When set, a multi-band
    /// compressor and bus limiter process the mixed audio of each chunk.
    dynamics_preset: Option<DynamicsPreset>,
    /// Persistent multi-band compressor. Stateful envelope followers carry
    /// across chunks for smooth dynamics tracking. Created lazily from
    /// `dynamics_preset` in `ensure_assembler()`.
    compressor: Option<MultibandCompressor>,
    /// Persistent bus limiter applied after the compressor as a safety
    /// ceiling. Created lazily from `dynamics_preset`.
    limiter: Option<BusLimiter>,
}

impl StreamingChorusSession {
    fn remaining_precompile_shapes(&self) -> Option<PrecompileShapes> {
        self.chunk_mode.remaining_precompile_shapes(self.cursor)
    }

    /// Create a new streaming chorus session.
    ///
    /// # Arguments
    ///
    /// * `chunks` - Pre-chunked token IDs as `DynTensor`s, each `[1, T_i]`.
    ///   All voices synthesize the same chunks (same text, different styles).
    /// * `styles` - Per-voice style embeddings `[1, 2*style_dim]`.
    /// * `speed` - Speaking rate multiplier.
    /// * `stream_config` - Crossfade configuration.
    ///
    /// # Errors
    ///
    /// Returns error if `stream_config` validation fails.
    pub fn new(
        chunks: Vec<DynTensor>,
        styles: Vec<DynTensor>,
        speed: f32,
        stream_config: KokoroStreamConfig,
    ) -> Result<Self, CompiledKokoroError> {
        stream_config.validate()?;
        Ok(Self {
            chunk_mode: ChorusChunkMode::SharedText(chunks),
            styles,
            speed,
            stream_config,
            cursor: 0,
            assembler: None,
            cancelled: false,
            precompile: true,
            precompiled: false,
            reverb_config: None,
            streaming_reverb: None,
            dynamics_preset: None,
            compressor: None,
            limiter: None,
        })
    }

    /// Create a new streaming chorus session where each voice synthesizes
    /// different text.
    ///
    /// # Arguments
    ///
    /// * `per_voice_chunks` - Per-voice chunk lists. `per_voice_chunks[v][i]`
    ///   is voice `v`'s chunk `i` (token IDs as `DynTensor` `[1, T]`).
    ///   All voices must have the same number of chunks.
    /// * `styles` - Per-voice style embeddings `[1, 2*style_dim]`.
    ///   Must have the same length as `per_voice_chunks`.
    /// * `speed` - Speaking rate multiplier.
    /// * `stream_config` - Crossfade configuration.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `per_voice_chunks` is empty.
    /// - Voices have different chunk counts.
    /// - `styles.len() != per_voice_chunks.len()`.
    /// - `stream_config` validation fails.
    pub fn new_varied_text(
        per_voice_chunks: Vec<Vec<DynTensor>>,
        styles: Vec<DynTensor>,
        speed: f32,
        stream_config: KokoroStreamConfig,
    ) -> Result<Self, CompiledKokoroError> {
        stream_config.validate()?;

        if per_voice_chunks.is_empty() {
            return Err(CompiledKokoroError::InvalidInput(
                "per_voice_chunks must not be empty".into(),
            ));
        }
        if styles.len() != per_voice_chunks.len() {
            return Err(CompiledKokoroError::InvalidInput(format!(
                "styles.len() ({}) must equal per_voice_chunks.len() ({})",
                styles.len(),
                per_voice_chunks.len(),
            )));
        }
        let expected_len = per_voice_chunks[0].len();
        for (i, voice_chunks) in per_voice_chunks.iter().enumerate().skip(1) {
            if voice_chunks.len() != expected_len {
                return Err(CompiledKokoroError::InvalidInput(format!(
                    "voice {} has {} chunks but voice 0 has {} — all voices \
                     must have equal chunk counts",
                    i,
                    voice_chunks.len(),
                    expected_len,
                )));
            }
        }

        Ok(Self {
            chunk_mode: ChorusChunkMode::PerVoice(per_voice_chunks),
            styles,
            speed,
            stream_config,
            cursor: 0,
            assembler: None,
            cancelled: false,
            precompile: true,
            precompiled: false,
            reverb_config: None,
            streaming_reverb: None,
            dynamics_preset: None,
            compressor: None,
            limiter: None,
        })
    }

    /// Enable spatial reverb on the streaming output.
    ///
    /// The reverb state persists across chunks, so the reverb tail from
    /// one chunk feeds naturally into the next. The `StreamingReverb`
    /// processor is lazily created on the first `next_chunk()` call
    /// (once the channel count is known from the `KokoroChorus` config).
    ///
    /// # Arguments
    ///
    /// * `config` - Reverb parameters (room size, mix, damping, etc.).
    ///   Use preset constructors like [`ReverbConfig::medium_hall()`] or
    ///   build a custom config via method chaining.
    #[must_use]
    pub fn with_reverb(mut self, config: ReverbConfig) -> Self {
        self.reverb_config = Some(config);
        self
    }

    /// Enable multi-band dynamics compression on the streaming chorus output.
    ///
    /// A 3-band compressor (Linkwitz-Riley crossover at 300 Hz / 4 kHz) and
    /// bus limiter are applied to each chunk's mixed audio **after** voice
    /// mixing and **before** crossfade assembly.
    ///
    /// The compressor and limiter are stateful -- their envelope followers
    /// persist across chunks, providing smooth dynamics tracking throughout
    /// the streaming session.
    ///
    /// # Presets
    ///
    /// - [`DynamicsPreset::Broadcast`] -- moderate compression, -1 dB ceiling.
    /// - [`DynamicsPreset::Gentle`] -- light compression for natural dynamics.
    /// - [`DynamicsPreset::Aggressive`] -- heavy compression for dense mixes.
    /// - [`DynamicsPreset::Mastering`] -- transparent limiting.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let session = StreamingChorusSession::new(chunks, styles, 1.0, config)?
    ///     .with_dynamics(DynamicsPreset::Broadcast)?;
    /// ```
    pub fn with_dynamics(mut self, preset: DynamicsPreset) -> Result<Self, CompiledKokoroError> {
        let comp_config = preset.to_config();
        self.compressor = Some(
            MultibandCompressor::new(&comp_config)
                .map_err(|e| CompiledKokoroError::InvalidInput(format!("dynamics config: {e}")))?,
        );
        let limiter = match preset {
            DynamicsPreset::Broadcast => BusLimiter::with_ceiling_db(-1.0),
            _ => BusLimiter::new(),
        };
        self.limiter = Some(limiter);
        self.dynamics_preset = Some(preset);
        Ok(self)
    }

    /// Set whether to precompile pipeline segments for the chunk shapes
    /// before the first synthesis.
    ///
    /// When `true` (the default), the first [`next_chunk`](Self::next_chunk)
    /// call runs one-time warmup for each chorus voice using shapes tailored
    /// to the token lengths present in this session. This reduces first-chunk
    /// latency for streaming chorus at the cost of an upfront warmup pass.
    ///
    /// Set to `false` for latency-sensitive cold starts where the caller
    /// prefers immediate first-chunk synthesis with compilation on demand.
    #[must_use]
    pub fn with_precompile(mut self, precompile: bool) -> Self {
        self.precompile = precompile;
        self
    }

    /// Whether shape precompilation is enabled for this session.
    #[must_use]
    pub fn precompile_enabled(&self) -> bool {
        self.precompile
    }

    /// Whether the next [`next_chunk`](Self::next_chunk) call will still run
    /// the session's one-time automatic warmup pass.
    ///
    /// This is `true` only when precompilation is enabled, warmup has not yet
    /// been consumed for this session, and the remaining unsynthesized chunk
    /// range yields at least one valid shape to warm.
    #[must_use]
    pub fn precompile_pending(&self) -> bool {
        self.precompile && !self.precompiled && self.remaining_precompile_shapes().is_some()
    }

    /// Whether the session's one-time warmup step has already been consumed.
    ///
    /// This becomes `true` after either explicit [`run_precompile`](Self::run_precompile)
    /// or the automatic first-chunk warmup path runs successfully enough to
    /// consume the warmup attempt for this session.
    #[must_use]
    pub fn precompile_consumed(&self) -> bool {
        self.precompiled
    }

    /// Inspect the exact shape set that automatic warmup would use for the
    /// remaining unsynthesized chunks.
    ///
    /// Returns `None` when automatic warmup is disabled, already consumed for
    /// this session, or there are no valid remaining chunk lengths to warm.
    #[must_use]
    pub fn pending_precompile_shapes(&self) -> Option<PrecompileShapes> {
        if !self.precompile_pending() {
            return None;
        }
        self.remaining_precompile_shapes()
    }

    /// Explicitly run shape precompilation for all chorus voices.
    ///
    /// Generates a tailored [`PrecompileShapes`] from the remaining
    /// unsynthesized chunk range and warms each voice once. Returns the
    /// total number of segment compilations reported across all voices, or 0
    /// if there are no chunk shapes to warm or precompilation already ran.
    pub fn run_precompile(
        &mut self,
        chorus: &mut KokoroChorus,
        cache: &PipelineCache,
    ) -> Result<usize, CompiledKokoroError> {
        if self.precompiled {
            return Ok(0);
        }
        self.precompiled = true;

        let shapes = match self.remaining_precompile_shapes() {
            Some(shapes) => shapes,
            None => return Ok(0),
        };

        let mut compiled = 0;
        for i in 0..chorus.n_voices() {
            let voice = chorus.voice_mut(i).ok_or_else(|| {
                CompiledKokoroError::InvalidInput(format!("chorus voice {i} not available"))
            })?;
            compiled += voice.warmup(&shapes, cache)?;
        }
        Ok(compiled)
    }

    /// Synthesize the next chunk. Returns `None` when all chunks are consumed
    /// or the session has been cancelled.
    ///
    /// Each call synthesizes one chunk across all voices using shared encoding,
    /// mixes the per-voice audio, applies crossfade with the previous chunk's
    /// tail, and returns a ready-to-play `AudioChunk`.
    ///
    /// # Arguments
    ///
    /// * `chorus` - The chorus pool (N voice instances with shared weights).
    /// * `cache` - Metal pipeline cache.
    ///
    /// # Returns
    ///
    /// - `Some(Ok(audio_chunk))` - Mixed, crossfaded audio for the current chunk.
    /// - `Some(Err(..))` - Synthesis/mixing failed. Cursor is still advanced.
    /// - `None` - All chunks consumed or session cancelled.
    pub fn next_chunk(
        &mut self,
        chorus: &mut KokoroChorus,
        cache: &PipelineCache,
    ) -> Option<Result<AudioChunk, CompiledKokoroError>> {
        if self.cancelled || self.cursor >= self.chunk_mode.len() {
            return None;
        }

        // Reset SineGen phase on all voices at the start of a new streaming
        // session so each voice begins with zero phase. Subsequent chunks
        // carry phase continuity automatically via build_harmonic_source.
        if self.cursor == 0 {
            for i in 0..chorus.n_voices() {
                if let Some(voice) = chorus.voice_mut(i) {
                    voice.reset_sinegen_phase();
                }
            }
        }

        // Run precompilation once before the first synthesis.
        if self.precompile && !self.precompiled {
            if let Err(err) = self.run_precompile(chorus, cache) {
                // Precompilation failure is non-fatal — synthesis will
                // compile on demand for any missed shapes.
                let _ = err;
            }
        }

        let idx = self.cursor;
        self.cursor += 1;

        let result = match &self.chunk_mode {
            ChorusChunkMode::SharedText(_) => self.synthesize_shared_chunk(idx, chorus, cache),
            ChorusChunkMode::PerVoice(_) => self.synthesize_per_voice_chunk(idx, chorus, cache),
        };
        Some(result)
    }

    /// Lazily initialize the `StreamingAssembler` and `StreamingReverb`
    /// on the first chunk, now that we know the channel count from the
    /// chorus config.
    fn ensure_assembler(
        &mut self,
        chorus: &KokoroChorus,
    ) -> Result<&mut StreamingAssembler, CompiledKokoroError> {
        if self.assembler.is_none() {
            let is_stereo = chorus.config().pans.is_some()
                || chorus.has_stereo()
                || chorus.has_chorus_pipeline();
            let channels: usize = if is_stereo { 2 } else { 1 };
            // For stereo, double crossfade_samples so the crossfade covers
            // the same time duration (interleaved stereo has 2 floats per
            // time-domain sample).
            let effective_config = if is_stereo {
                KokoroStreamConfig::new(self.stream_config.crossfade_samples * 2)?
                    .with_window(self.stream_config.crossfade_window)
            } else {
                self.stream_config.clone()
            };
            let asm = StreamingAssembler::new_with_channels(
                effective_config,
                self.chunk_mode.len(),
                channels,
            )?;
            self.assembler = Some(asm);

            // Initialize streaming reverb if configured. The reverb
            // processor persists across chunks so the delay-line state
            // (reverb tail) carries from one chunk to the next.
            if let Some(ref config) = self.reverb_config {
                self.streaming_reverb = Some(StreamingReverb::new(config.clone(), is_stereo)?);
            }
        }
        Ok(self
            .assembler
            .as_mut()
            .expect("invariant: assembler set above"))
    }

    /// Internal: synthesize one shared-text chunk across all voices, mix, and
    /// crossfade. Steps 1-2 are shared (encode once), Steps 3-8 run per voice.
    fn synthesize_shared_chunk(
        &mut self,
        idx: usize,
        chorus: &mut KokoroChorus,
        cache: &PipelineCache,
    ) -> Result<AudioChunk, CompiledKokoroError> {
        let chunk_input = match &self.chunk_mode {
            ChorusChunkMode::SharedText(chunks) => &chunks[idx],
            ChorusChunkMode::PerVoice(_) => {
                return Err(CompiledKokoroError::InvalidInput(
                    "synthesize_shared_chunk called on PerVoice mode".into(),
                ));
            }
        };
        let n = chorus.n_voices();
        let max_pos = chorus
            .voice(0)
            .ok_or_else(|| CompiledKokoroError::InvalidInput("chorus has no voices".into()))?
            .config()
            .plbert
            .max_position_embeddings;

        validate_input_ids(chunk_input, max_pos)?;

        // Reclaim buffer pool entries.
        crate::arena::pool_reclaim();

        // Pre-size the arena for this chunk's N-voice decode loop (#4289).
        // Multiply by 2 to cover shared encode plus one voice's decode work.
        let arena_estimate = chorus
            .voice(0)
            .expect("voice 0 already validated above")
            .estimate_arena_bytes()
            .saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        // Pre-split styles and upload to GPU.
        let v0 = chorus
            .voice(0)
            .ok_or_else(|| CompiledKokoroError::InvalidInput("chorus has no voices".into()))?;
        let mut decoder_styles: Vec<DynTensor> = Vec::with_capacity(n);
        let mut prosody_styles: Vec<DynTensor> = Vec::with_capacity(n);
        for style in &self.styles {
            let split = v0.split_style(style)?;
            decoder_styles.push(split.decoder_style.to_device(&gpu())?);
            prosody_styles.push(split.prosody_style.to_device(&gpu())?);
        }

        let result = (|| -> Result<Vec<Vec<f32>>, CompiledKokoroError> {
            let chunk_audios: Vec<DynTensor> = with_nan_check_policy(
                NanCheckPolicy::Skip,
                || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                    // Steps 1-2: encode once on voice[0].
                    let enc = chorus
                        .voice_mut(0)
                        .ok_or_else(|| {
                            CompiledKokoroError::InvalidInput("chorus has no voices".into())
                        })?
                        .step_encode(chunk_input, cache)?;

                    // Steps 3-8 per voice using shared encoding.
                    let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                    for i in 0..n {
                        let voice = chorus.voice_mut(i).ok_or_else(|| {
                            CompiledKokoroError::InvalidInput(format!(
                                "chorus voice {i} not available"
                            ))
                        })?;
                        let _arena_cp = DefaultArenaCheckpoint::new();
                        audios.push(run_voice_pipeline(
                            voice,
                            &enc,
                            &prosody_styles[i],
                            &decoder_styles[i],
                            self.speed,
                            cache,
                        )?);
                    }
                    Ok(audios)
                },
            )?;

            // Outside NaN-skip scope: verify + extract PCM per voice.
            verify_and_extract_pcm(&chorus.voices, &chunk_audios)
        })();

        self.mix_and_crossfade(result, chorus, cache)
    }

    /// Internal: synthesize one per-voice chunk. Each voice runs the full
    /// pipeline independently (Steps 1-8) with its own chunk token IDs.
    fn synthesize_per_voice_chunk(
        &mut self,
        idx: usize,
        chorus: &mut KokoroChorus,
        cache: &PipelineCache,
    ) -> Result<AudioChunk, CompiledKokoroError> {
        let n = chorus.n_voices();
        let max_pos = chorus
            .voice(0)
            .ok_or_else(|| CompiledKokoroError::InvalidInput("chorus has no voices".into()))?
            .config()
            .plbert
            .max_position_embeddings;

        // Validate all per-voice chunks for this index.
        let per_voice = match &self.chunk_mode {
            ChorusChunkMode::PerVoice(pv) => pv,
            ChorusChunkMode::SharedText(_) => {
                return Err(CompiledKokoroError::InvalidInput(
                    "synthesize_per_voice_chunk called on SharedText mode".into(),
                ));
            }
        };
        for (v, voice_chunks) in per_voice.iter().enumerate() {
            if idx >= voice_chunks.len() {
                return Err(CompiledKokoroError::InvalidInput(format!(
                    "voice {v} has no chunk at index {idx}"
                )));
            }
            validate_input_ids(&voice_chunks[idx], max_pos)?;
        }

        // Reclaim buffer pool entries.
        crate::arena::pool_reclaim();

        // Pre-size the arena for this chunk's N-voice full pipeline loop
        // (#4289). Multiply by 2 to cover one voice's encode plus decode work.
        let arena_estimate = chorus
            .voice(0)
            .expect("voice 0 already validated above")
            .estimate_arena_bytes()
            .saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        // Pre-split styles and upload to GPU.
        let v0 = chorus
            .voice(0)
            .ok_or_else(|| CompiledKokoroError::InvalidInput("chorus has no voices".into()))?;
        let mut decoder_styles: Vec<DynTensor> = Vec::with_capacity(n);
        let mut prosody_styles: Vec<DynTensor> = Vec::with_capacity(n);
        for style in &self.styles {
            let split = v0.split_style(style)?;
            decoder_styles.push(split.decoder_style.to_device(&gpu())?);
            prosody_styles.push(split.prosody_style.to_device(&gpu())?);
        }

        let result = (|| -> Result<Vec<Vec<f32>>, CompiledKokoroError> {
            let chunk_audios: Vec<DynTensor> = with_nan_check_policy(
                NanCheckPolicy::Skip,
                || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                    let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                    for i in 0..n {
                        // Each voice runs its own encode (Steps 1-2).
                        let voice_chunk = match &self.chunk_mode {
                            ChorusChunkMode::PerVoice(pv) => &pv[i][idx],
                            _ => unreachable!(),
                        };
                        let _arena_cp = DefaultArenaCheckpoint::new();
                        let voice = chorus.voice_mut(i).ok_or_else(|| {
                            CompiledKokoroError::InvalidInput(format!(
                                "chorus voice {i} not available"
                            ))
                        })?;
                        let enc = voice.step_encode(voice_chunk, cache)?;

                        // Steps 3-8 with this voice's encoding.
                        audios.push(run_voice_pipeline(
                            voice,
                            &enc,
                            &prosody_styles[i],
                            &decoder_styles[i],
                            self.speed,
                            cache,
                        )?);
                    }
                    Ok(audios)
                },
            )?;

            // Outside NaN-skip scope: verify + extract PCM per voice.
            verify_and_extract_pcm(&chorus.voices, &chunk_audios)
        })();

        self.mix_and_crossfade(result, chorus, cache)
    }

    /// Shared tail: mix voice PCMs and apply crossfade. Used by both
    /// `synthesize_shared_chunk` and `synthesize_per_voice_chunk`.
    ///
    /// Routes through the chorus's full processing chain via
    /// [`KokoroChorus::mix_or_process`]: detune, humanize, per-voice
    /// EQ/de-essing, stereo/mono mix, bus EQ, dynamics (or the
    /// [`ChorusMasterPipeline`](nn_models::kokoro_chorus_pipeline::ChorusMasterPipeline)
    /// when configured). Processing state on `chorus` persists across
    /// chunks for smooth dynamics tracking.
    ///
    /// Session-level dynamics (from [`with_dynamics`](Self::with_dynamics))
    /// are applied as a secondary stage only when the chorus itself has no
    /// dynamics or pipeline configured, preventing double-compression.
    fn mix_and_crossfade(
        &mut self,
        result: Result<Vec<Vec<f32>>, CompiledKokoroError>,
        chorus: &mut KokoroChorus,
        cache: &PipelineCache,
    ) -> Result<AudioChunk, CompiledKokoroError> {
        let _ = cache; // retained for signature consistency
        let mut voice_pcms = match result {
            Ok(v) => v,
            Err(e) => {
                crate::gpu_scope::discard_pending_batch();
                return Err(e);
            }
        };

        // Route through the full chorus processing chain: detune, humanize,
        // per-voice EQ/de-essing, stereo/mono mix, bus EQ, dynamics (or
        // ChorusMasterPipeline when configured). This matches the
        // non-streaming synthesize_chorus path exactly.
        let mut mixed = chorus.mix_or_process(&mut voice_pcms)?;

        // Session-level dynamics (fallback). Only applied when the chorus
        // itself does not have dynamics or a ChorusMasterPipeline configured,
        // since those already include dynamics processing. This prevents
        // double-compression while still supporting session-only dynamics
        // via `with_dynamics()`.
        if !chorus.has_dynamics() && !chorus.has_chorus_pipeline() {
            if let Some(ref mut comp) = self.compressor {
                comp.process(&mut mixed);
            }
            if let Some(ref mut lim) = self.limiter {
                lim.process(&mut mixed);
            }
        }

        // Apply streaming reverb (persistent state across chunks).
        // Lazily initialized in ensure_assembler on the first chunk.
        let _ = self.ensure_assembler(chorus)?;
        if let Some(ref mut reverb) = self.streaming_reverb {
            reverb.process_chunk(&mut mixed);
        }

        // Crossfade via the shared StreamingAssembler.
        let assembler = self
            .assembler
            .as_mut()
            .expect("invariant: assembler initialized by ensure_assembler");
        let audio_chunk = assembler.push_raw(mixed)?;

        Ok(audio_chunk)
    }

    /// Whether this session uses per-voice text (each voice synthesizes
    /// different text).
    #[must_use]
    pub fn is_varied_text(&self) -> bool {
        matches!(self.chunk_mode, ChorusChunkMode::PerVoice(_))
    }

    /// Number of remaining chunks (not yet synthesized).
    #[must_use]
    pub fn remaining(&self) -> usize {
        if self.cancelled {
            return 0;
        }
        self.chunk_mode.len().saturating_sub(self.cursor)
    }

    /// Whether all chunks have been synthesized or the session was cancelled.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.cancelled || self.cursor >= self.chunk_mode.len()
    }

    /// Total number of chunks in this session.
    #[must_use]
    pub fn total_chunks(&self) -> usize {
        self.chunk_mode.len()
    }

    /// Number of chunks already synthesized.
    #[must_use]
    pub fn synthesized_count(&self) -> usize {
        self.cursor
    }

    /// Cancel the session. Subsequent `next_chunk()` calls return `None`.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        // Release crossfade, reverb, and dynamics state.
        self.assembler = None;
        self.streaming_reverb = None;
        if let Some(ref mut comp) = self.compressor {
            comp.reset();
        }
        if let Some(ref mut lim) = self.limiter {
            lim.reset();
        }
    }

    /// Whether the session has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Reset the session to re-synthesize from the beginning.
    ///
    /// Clears crossfade state, resets cursor and sample offset.
    /// Useful for re-synthesizing the same text with different voice
    /// styles without re-tokenizing.
    pub fn reset(&mut self) {
        self.cursor = 0;
        // Reset the assembler and reverb state if initialized; drop them
        // so they get re-created with the correct channel count on the
        // next call. This clears the reverb delay lines so a previous
        // session's tail does not bleed into the new session.
        self.assembler = None;
        self.streaming_reverb = None;
        self.cancelled = false;
        // Reset dynamics envelope followers so a previous session's
        // gain reduction state does not affect the new session.
        if let Some(ref mut comp) = self.compressor {
            comp.reset();
        }
        if let Some(ref mut lim) = self.limiter {
            lim.reset();
        }
    }

    /// Update the speaking rate for subsequent chunks.
    ///
    /// Already-synthesized chunks are not affected.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Current speaking rate.
    #[must_use]
    pub fn speed(&self) -> f32 {
        self.speed
    }
}

#[cfg(test)]
#[path = "compiled_kokoro_chorus_pull_streaming_tests.rs"]
mod tests;

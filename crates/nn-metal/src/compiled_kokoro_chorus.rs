// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Multi-voice chorus pipeline for GPU-accelerated Kokoro TTS.
//!
//! [`KokoroChorus`] manages N voice instances created via
//! [`CompiledKokoro::clone_dispatch()`], each sharing model weights through
//! `Arc`. Synthesis runs sequentially on the calling thread (CompiledKokoro
//! is `!Send`), but GPU work from consecutive voices is pipelined by Metal.
//!
//! # Architecture
//!
//! **Batch mode** (full utterance per voice, then mix):
//! ```text
//! KokoroChorus::new(primary, ChorusConfig::equal_gain(8)?)
//!   → 8 CompiledKokoro instances (Arc-shared weights)
//!   → synthesize_chorus(inputs, styles, speed, cache)
//!   → 8 × synthesize()  (sequential, GPU-pipelined)
//!   → mix_voices_with_config()  (mono or stereo based on pans)
//!   → Vec<f32>           (final mixed PCM, interleaved stereo if pans set)
//! ```
//!
//! **Streaming mode** (per-chunk mixing for lower first-audio latency):
//! ```text
//! KokoroChorus::synthesize_streaming_chorus(chunks, styles, speed, stream_config, cache)
//!   → for each chunk: N × synthesize → mix → AudioChunk
//!   → crossfade at chunk boundaries
//!   → Vec<AudioChunk>    (playable as each chunk completes)
//! ```
//!
//! First-audio latency: 1 chunk × N voices (not full utterance × N voices).
//!
//! Memory: 8 voices use ~1.02x the GPU memory of 1 (weight buffers are
//! aliased via `MetalBuffer::alias()`, only segment caches are per-voice).
//!
//! Part of #3355, #3351, #2740.

use nn_core::dyn_tensor::DynTensor;
use nn_models::kokoro_chorus::{mix_voices_from_refs, ChorusConfig};
use nn_models::kokoro_chorus_adaptive_dynamics::AdaptiveDynamicsConfig;
use nn_models::kokoro_chorus_alignment::{align_voices, AlignmentConfig};
use nn_models::kokoro_chorus_auto_eq::AutoEqConfig;
use nn_models::kokoro_chorus_automation::MixAutomator;
use nn_models::kokoro_chorus_bleed::{apply_voice_bleed, BleedConfig};
use nn_models::kokoro_chorus_breath::{
    detect_pauses, insert_breath_sounds, BreathConfig, BreathGenerator,
};
use nn_models::kokoro_chorus_character::CharacterConfig;
use nn_models::kokoro_chorus_convolution::ConvolutionConfig;
use nn_models::kokoro_chorus_crossfade::{CrossfadeOptimizer, CrossfadeOptimizerConfig};
use nn_models::kokoro_chorus_detune::{apply_detune, cents_to_rate, DetuneConfig};
use nn_models::kokoro_chorus_dither::DitherConfig;
use nn_models::kokoro_chorus_doubler::{apply_doubler_per_voice, DoublerConfig};
use nn_models::kokoro_chorus_ducking::{DuckingConfig, SpectralDucker};
use nn_models::kokoro_chorus_dynamics::{BusLimiter, DynamicsPreset, MultibandCompressor};
use nn_models::kokoro_chorus_ensemble::EnsembleConfig;
use nn_models::kokoro_chorus_exciter::{apply_exciter, ExciterConfig};
use nn_models::kokoro_chorus_formant::{
    shift_pitch_preserve_formant, simple_pitch_shift, FormantPreserveConfig,
};
use nn_models::kokoro_chorus_formant_tune::FormantTuneConfig;
use nn_models::kokoro_chorus_freeze::FreezeConfig;
use nn_models::kokoro_chorus_gain_staging::GainStagingConfig;
use nn_models::kokoro_chorus_gate::GateConfig;
use nn_models::kokoro_chorus_hrtf::HrtfConfig;
use nn_models::kokoro_chorus_humanize::{apply_humanize, HumanizeConfig};
use nn_models::kokoro_chorus_intonation::IntonationConfig;
use nn_models::kokoro_chorus_loudness::LoudnessConfig;
use nn_models::kokoro_chorus_micro_pitch::MicroPitchConfig;
use nn_models::kokoro_chorus_multiband_stereo::MultibandStereoConfig;
use nn_models::kokoro_chorus_onset_sync::OnsetSyncConfig;
use nn_models::kokoro_chorus_oversample::OversampleConfig;
use nn_models::kokoro_chorus_pipeline::{ChorusMasterConfig, ChorusMasterPipeline};
use nn_models::kokoro_chorus_pitch_correct::{apply_pitch_correction, PitchCorrectConfig};
use nn_models::kokoro_chorus_room::RoomConfig;
use nn_models::kokoro_chorus_saturation::{SaturationConfig, SaturationProcessor};
use nn_models::kokoro_chorus_sibilance::SibilanceConfig;
use nn_models::kokoro_chorus_spatial::{
    auto_layout_spatial, process_voice_spatial, SpatialConfig,
};
use nn_models::kokoro_chorus_spectral_match::SpectralMatchConfig;
use nn_models::kokoro_chorus_stereo_analysis::StereoAnalysisConfig;
use nn_models::kokoro_chorus_sub_bass::SubBassConfig;
use nn_models::kokoro_chorus_tilt::TiltConfig;
use nn_models::kokoro_chorus_transient::{apply_transient_shaping, TransientConfig};
use nn_models::kokoro_chorus_warmth::WarmthConfig;
use nn_models::kokoro_chorus_width::StereoWidthConfig;

use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_models::kokoro_chorus_eq::{MixBusConfig, MixBusProcessor};
use nn_models::kokoro_chorus_stereo::{apply_stereo_mix, interleave_stereo, StereoChorusConfig};
use nn_models::kokoro_error::{validate_speed, KokoroError};

use super::segment_pipeline::run_decode_phase;
use super::{gpu, validate_input_ids, CompiledKokoro, CompiledKokoroError, StepEncodeResult};
use crate::cache::PipelineCache;
use crate::gpu_fence::GpuFence;

/// Run Steps 3-8 (prosody -> regulate -> f0/energy -> harmonic -> generate -> iSTFT)
/// for a single voice using shared encoding results.
///
/// Free function to avoid `&mut self` borrowing conflicts when iterating
/// over `voices` while calling methods on individual voice instances.
///
/// Uses the two-phase pattern: Phase 1 (prosody + regulate with GPU sync),
/// then submits via GpuFence, then Phase 2 (f0 -> harmonic -> generate ->
/// iSTFT, sync-free) with per-segment fences for CPU-GPU overlap (#4264).
pub(crate) fn run_voice_pipeline(
    voice: &mut CompiledKokoro,
    enc: &StepEncodeResult,
    prosody_style: &DynTensor,
    decoder_style: &DynTensor,
    speed: f32,
    cache: &PipelineCache,
) -> Result<DynTensor, CompiledKokoroError> {
    // Phase 1: prosody + regulate (has GPU sync for prefix-sum readback).
    let pros = voice.step_predict_prosody(&enc.bert_features, prosody_style, enc.seq_len, cache)?;
    let reg = voice.step_regulate(
        &pros.dur_logits,
        &pros.features,
        &enc.text_features,
        speed,
        cache,
    )?;

    // Submit Phase 1 GPU work non-blocking. Metal queue ordering guarantees
    // Phase 1 work completes before Phase 2 work executes on GPU.
    let _fence_phase1 =
        GpuFence::submit_current().map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;

    // Phase 2: f0 -> harmonic -> generate -> iSTFT with per-segment fences.
    run_decode_phase(voice, &reg, prosody_style, decoder_style, cache)
}

/// Run Steps 5-8 (f0/energy → harmonic → generate → iSTFT) for a single voice
/// using a pre-computed regulate result.
///
/// Used by the batch-regulate chorus path (#4290) to separate the sync-heavy
/// regulate phase from the sync-free decode phase, enabling GPU pipelining
/// across voices.
pub(crate) fn run_voice_decode(
    voice: &mut CompiledKokoro,
    reg: &super::StepRegulateResult,
    prosody_style: &DynTensor,
    decoder_style: &DynTensor,
    cache: &PipelineCache,
) -> Result<DynTensor, CompiledKokoroError> {
    let f0e = voice.step_predict_f0_energy(&reg.aligned_dur, prosody_style, reg.t_mel, cache)?;
    let har = voice.step_harmonic_source(&f0e.f0, &f0e.energy, reg.t_mel, cache)?;
    let generator = voice.step_generate(
        &reg.regulated,
        &f0e.f0,
        &f0e.energy,
        decoder_style,
        &har,
        reg.t_mel,
        cache,
    )?;
    voice.step_istft(&generator.magnitude, &generator.phase, cache)
}

/// Run voice decode (steps 5-8) without synchronizing the GPU.
///
/// Submits the voice's post-regulate GPU commands as a non-blocking
/// [`GpuFence`] after encoding steps 5-8 into the command buffer.
/// The caller collects fences from all voices and waits on them in bulk,
/// allowing the GPU to execute multiple voices' decode work concurrently
/// across different command buffers.
///
/// Returns the audio `DynTensor` (GPU-resident, not yet readable) and a
/// `GpuFence` handle. The tensor data is only valid after `fence.wait()`.
///
/// # Errors
///
/// Returns error if any decode step fails or if the GPU fence submission fails.
///
/// Part of #4290.
pub(crate) fn run_voice_decode_async(
    voice: &mut CompiledKokoro,
    reg: &super::StepRegulateResult,
    prosody_style: &DynTensor,
    decoder_style: &DynTensor,
    cache: &PipelineCache,
) -> Result<(DynTensor, Option<GpuFence>), CompiledKokoroError> {
    let audio = run_voice_decode(voice, reg, prosody_style, decoder_style, cache)?;

    // Submit the pending GPU work non-blocking.
    let fence = GpuFence::submit_current().map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;

    Ok((audio, fence))
}

/// NaN guard + verify + flatten to PCM for each voice's audio tensor.
///
/// Validates all audio tensors outside the NaN-skip scope, extracts flat
/// `Vec<f32>` PCM buffers, and runs `step_verify` on each.
pub(crate) fn verify_and_extract_pcm(
    voices: &[CompiledKokoro],
    audio_tensors: &[DynTensor],
) -> Result<Vec<Vec<f32>>, CompiledKokoroError> {
    let mut out = Vec::with_capacity(audio_tensors.len());
    for (i, audio) in audio_tensors.iter().enumerate() {
        // GPU→CPU transfer: single flush commits all pending GPU work for this voice.
        let audio = audio
            .to_device(&super::cpu())
            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
        let pcm = extract_finite_pcm_from_audio(&audio, &format!("chorus_voice_{i}_audio"))?;
        let _cert = voices[i].step_verify(&audio)?;
        out.push(pcm);
    }
    Ok(out)
}

/// Extract PCM samples from an audio tensor with a zero-copy CPU-F32 fast path.
///
/// `DynTensor::to_flat_vec::<f32>()` clones CPU F32 tensors into an owned
/// `ArrayD<f32>` before collecting. Chorus hot paths already move audio to CPU
/// for verification, so when the tensor is CPU F32 we can collect directly from
/// the borrowed view and avoid that extra full-buffer allocation/copy.
///
/// Falls back to the generic flattening path for non-F32 tensors to preserve
/// existing behavior.
pub(crate) fn extract_pcm_from_audio(audio: &DynTensor) -> Result<Vec<f32>, CompiledKokoroError> {
    if audio.device().is_gpu() {
        let cpu_audio = audio
            .to_device(&super::cpu())
            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
        return extract_pcm_from_audio(&cpu_audio);
    }

    match audio.as_cpu_f32() {
        Ok(view) => Ok(view.iter().copied().collect()),
        Err(_) => audio
            .to_flat_vec::<f32>()
            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e))),
    }
}

/// Extract PCM samples while validating that all values are finite.
///
/// This keeps the common CPU-F32 chorus path to a single pass over the audio
/// buffer instead of a separate `any_non_finite()` scan followed by PCM
/// extraction.
pub(crate) fn extract_finite_pcm_from_audio(
    audio: &DynTensor,
    name: &str,
) -> Result<Vec<f32>, CompiledKokoroError> {
    if audio.device().is_gpu() {
        let cpu_audio = audio
            .to_device(&super::cpu())
            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
        return extract_finite_pcm_from_audio(&cpu_audio, name);
    }

    match audio.as_cpu_f32() {
        Ok(view) => {
            let mut pcm = Vec::with_capacity(view.len());
            let mut non_finite_count = 0usize;
            for &sample in view.iter() {
                if sample.is_finite() {
                    if non_finite_count == 0 {
                        pcm.push(sample);
                    }
                } else {
                    non_finite_count += 1;
                }
            }
            if non_finite_count > 0 {
                return Err(nn_core::TensorError::NonFiniteData {
                    name: name.to_string(),
                    count: non_finite_count,
                }
                .into());
            }
            Ok(pcm)
        }
        Err(_) => {
            let pcm = audio
                .to_flat_vec::<f32>()
                .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
            let non_finite_count = pcm.iter().filter(|sample| !sample.is_finite()).count();
            if non_finite_count > 0 {
                return Err(nn_core::TensorError::NonFiniteData {
                    name: name.to_string(),
                    count: non_finite_count,
                }
                .into());
            }
            Ok(pcm)
        }
    }
}

/// Apply adaptive alignment to per-voice PCM buffers without pre-cloning them.
///
/// `align_voices` already allocates its own output buffers for the aligned
/// result. The chorus hot path only needs the input slices by immutable
/// reference, so cloning `voice_audio` before the call is redundant.
pub(crate) fn apply_alignment_in_place(
    voice_audio: &mut [Vec<f32>],
    config: &AlignmentConfig,
) -> Result<(), CompiledKokoroError> {
    config
        .validate()
        .map_err(|e| CompiledKokoroError::InvalidInput(format!("alignment: {e}")))?;

    if voice_audio.len() <= 1
        || voice_audio.first().map_or(true, Vec::is_empty)
        || config.tightness <= 0.0
    {
        return Ok(());
    }

    let aligned = align_voices(voice_audio, config)
        .map_err(|e| CompiledKokoroError::InvalidInput(format!("alignment: {e}")))?;
    for (dst, src) in voice_audio.iter_mut().zip(aligned) {
        *dst = src;
    }
    Ok(())
}

/// Kokoro sample rate (24kHz).
pub(crate) const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Apply humanization to per-voice PCM buffers in place.
///
/// Each voice is independently humanized using its voice index as the
/// deterministic seed. This produces different breathing patterns, onset
/// jitter, micro-timing drift, and amplitude envelopes per voice while
/// keeping the output reproducible.
///
/// Called after `verify_and_extract_pcm` and before `mix_voices_with_config`.
pub(crate) fn humanize_voice_pcms(
    voice_audio: &mut [Vec<f32>],
    config: &HumanizeConfig,
) -> Result<(), CompiledKokoroError> {
    for (voice_index, pcm) in voice_audio.iter_mut().enumerate() {
        apply_humanize(pcm, config, voice_index, KOKORO_SAMPLE_RATE).map_err(|e| {
            CompiledKokoroError::InvalidInput(format!("humanize voice {voice_index}: {e}"))
        })?;
    }
    Ok(())
}

#[inline]
fn can_use_simple_mono_mix(config: &ChorusConfig) -> bool {
    config.pans.is_none()
        && config.pitch_semitones.is_none()
        && config.timing_offsets_sec.is_none()
        && config.soft_limiter_drive.is_none()
        && config.reverb.is_none()
}

/// Fast path for the common mono chorus mix configuration.
///
/// This matches the mono branch of `nn_models::kokoro_chorus::mix_voices_from_refs`
/// when no pitch offsets, timing offsets, soft limiter, stereo pans, or reverb
/// are configured. Keeping this local avoids building a temporary `Vec<&[f32]>`
/// and bypasses the generic feature-dispatch branch in the hot path.
fn mix_voice_audio_mono_simple(
    voice_audio: &[Vec<f32>],
    config: &ChorusConfig,
) -> Result<Vec<f32>, CompiledKokoroError> {
    config
        .validate()
        .map_err(|e| CompiledKokoroError::InvalidInput(format!("chorus config: {e}")))?;

    if voice_audio.len() != config.n_voices {
        return Err(CompiledKokoroError::InvalidInput(format!(
            "voice_audio length {} != config.n_voices {}",
            voice_audio.len(),
            config.n_voices,
        )));
    }
    if voice_audio.is_empty() {
        return Ok(Vec::new());
    }

    let max_len = voice_audio.iter().map(Vec::len).max().unwrap_or(0);
    if max_len == 0 {
        return Ok(Vec::new());
    }

    let mut mixed = vec![0.0f32; max_len];
    for (pcm, &gain) in voice_audio.iter().zip(config.gains.iter()) {
        let g = gain.clamp(0.0, 1.0);
        for (i, &sample) in pcm.iter().enumerate() {
            mixed[i] += sample * g;
        }
    }

    if config.clip_output {
        for sample in &mut mixed {
            *sample = sample.clamp(-1.0, 1.0);
        }
    }

    Ok(mixed)
}

/// Apply per-voice detuning to PCM buffers in place.
///
/// Each voice gets a different pitch offset (in cents) based on its index
/// and the detune configuration. Voice 0 is always left unmodified as the
/// anchor voice. Allpass Thiran interpolation preserves high-frequency
/// content during the fractional-sample resampling.
///
/// Called after `humanize_voice_pcms` and before `mix_voices_with_config`.
pub(crate) fn detune_voice_pcms(
    voice_audio: &mut [Vec<f32>],
    config: &DetuneConfig,
) -> Result<(), CompiledKokoroError> {
    apply_detune(voice_audio, config, KOKORO_SAMPLE_RATE)
        .map_err(|e| CompiledKokoroError::InvalidInput(format!("detune: {e}")))?;
    Ok(())
}

pub(crate) fn contextualize_invalid_input(
    err: CompiledKokoroError,
    context: impl std::fmt::Display,
) -> CompiledKokoroError {
    match err {
        CompiledKokoroError::InvalidInput(msg) => {
            CompiledKokoroError::InvalidInput(format!("{context}: {msg}"))
        }
        other => other,
    }
}

/// Scoped checkpoint for the thread-local default arena.
///
/// Restores the arena bump pointer when dropped, even if the caller returns
/// early or unwinds. This keeps chorus synthesis loops from leaking temporary
/// arena state on mid-voice failures.
pub(super) struct DefaultArenaCheckpoint(Option<(usize, u64)>);

impl DefaultArenaCheckpoint {
    #[inline]
    pub(super) fn new() -> Self {
        Self(crate::arena::checkpoint_default_arena())
    }
}

impl Drop for DefaultArenaCheckpoint {
    fn drop(&mut self) {
        crate::arena::restore_default_arena(self.0.take());
    }
}

pub(crate) fn validate_indexed_input_ids(
    inputs: &[DynTensor],
    max_position_embeddings: usize,
    label: &str,
) -> Result<(), CompiledKokoroError> {
    for (i, input_ids) in inputs.iter().enumerate() {
        validate_input_ids(input_ids, max_position_embeddings)
            .map_err(|err| contextualize_invalid_input(err, format!("{label}[{i}]")))?;
    }
    Ok(())
}

/// Multi-voice chorus synthesis pool using GPU-compiled Kokoro.
///
/// Each voice is an independent [`CompiledKokoro`] instance sharing model
/// weights via `Arc<SharedKokoroState>`. Use this for singing chorus,
/// multi-speaker dialogue, or ensemble effects.
///
/// # Example
///
/// ```rust,no_run
/// use nn_metal::compiled_kokoro::chorus::KokoroChorus;
/// use nn_models::kokoro_chorus::ChorusConfig;
///
/// let mut chorus = KokoroChorus::new(&primary_kokoro, ChorusConfig::equal_gain(8)?)?;
/// let mixed_audio = chorus.synthesize_chorus(&inputs, &styles, 1.0, &cache)?;
/// ```
pub struct KokoroChorus {
    /// Voice instances, each with independent segment caches but shared weights.
    pub(crate) voices: Vec<CompiledKokoro>,
    /// Chorus configuration (gains, clipping).
    pub(crate) config: ChorusConfig,
    /// Optional dynamics processing preset. When set, a multi-band compressor
    /// and bus limiter are applied to the mixed output of every synthesis call.
    dynamics_preset: Option<DynamicsPreset>,
    /// Persistent multi-band compressor (stateful: envelope followers carry
    /// across synthesis calls). Created lazily from `dynamics_preset`.
    compressor: Option<MultibandCompressor>,
    /// Persistent bus limiter applied after the compressor as a safety ceiling.
    /// Created lazily from `dynamics_preset`.
    limiter: Option<BusLimiter>,
    /// Optional adaptive voice alignment (cross-correlation temporal sync).
    /// When `Some`, voices are aligned to a common temporal reference before
    /// any other processing. Applied FIRST to fix timing before detuning/EQ.
    alignment_config: Option<AlignmentConfig>,
    /// Optional per-voice humanization (breathing, micro-timing, amplitude envelope).
    /// When `Some`, each voice's PCM audio is humanized before mixing.
    pub(crate) humanize_config: Option<HumanizeConfig>,
    /// Optional per-voice detuning configuration for pitch variation.
    /// When `Some`, each voice's PCM audio is resampled at a slightly different
    /// rate (allpass Thiran interpolation) before mixing, creating natural
    /// beating frequencies for a warm ensemble sound.
    detune_config: Option<DetuneConfig>,
    /// Optional bus saturation configuration. When `Some`, harmonic warmth is
    /// applied after dynamics and before any further processing in the default
    /// mixing path.
    saturation_config: Option<SaturationConfig>,
    /// Persistent saturation processor (stateful: decimation filter state).
    /// Created lazily from `saturation_config`.
    saturation_processor: Option<SaturationProcessor>,
    /// Optional integrated chorus master pipeline. When `Some`, replaces the
    /// default `mix_voices_with_config` + `apply_dynamics` path with the full
    /// ChorusMasterPipeline processing chain (detuning, EQ, de-essing, humanize,
    /// blend, stereo imaging, multiband dynamics, reverb, limiter).
    chorus_pipeline: Option<ChorusMasterPipeline>,
    /// Optional stereo imaging configuration. When `Some`, voices are panned
    /// using constant-power sin/cos panning and the output is interleaved
    /// stereo. When `None`, the output remains mono.
    stereo_config: Option<StereoChorusConfig>,
    /// Optional mix bus processor (per-voice EQ + de-esser + bus EQ).
    /// Created from `mix_bus_config` when `with_eq_config` is called.
    mix_bus: Option<MixBusProcessor>,
    /// Stored MixBusConfig for re-creating the processor if voice count changes.
    mix_bus_config: Option<MixBusConfig>,
    /// Optional formant-preserving pitch shift config for detuning.
    /// When set alongside `detune_config`, voices with large detuning (>10
    /// cents) use PSOLA formant-preserving pitch shift instead of basic allpass
    /// resampling, preventing formant distortion on heavily detuned voices.
    formant_config: Option<FormantPreserveConfig>,
    /// Optional per-voice breath noise configuration. When `Some`, synthetic
    /// breath sounds are inserted at detected pause regions with per-voice
    /// timing stagger. Applied after humanization and before mixing.
    breath_config: Option<BreathConfig>,
    /// Per-voice breath generator state (PRNG + lowpass filter per voice).
    /// Created lazily from `breath_config` when `with_breath` is called.
    breath_generator: Option<BreathGenerator>,
    /// Optional per-voice spatial depth processing. When `Some`, each voice
    /// is processed through distance attenuation, air absorption, propagation
    /// delay, and ILD stereo panning before mixing. Applied in `mix_or_process`
    /// when no chorus pipeline is active.
    spatial_config: Option<SpatialConfig>,
    /// Optional per-voice transient shaping configuration. When `Some`,
    /// transient shaping is applied after alignment, before detune in the
    /// default mixing path. Shapes consonant attacks and vowel sustain.
    transient_config: Option<TransientConfig>,
    /// Optional per-voice bleed (microphone crosstalk) configuration.
    /// When `Some`, voice bleed is applied after breath, before spatial/mix
    /// in the default mixing path.
    bleed_config: Option<BleedConfig>,
    /// Optional stereo width enhancement configuration. Stored for use
    /// with the ChorusMasterPipeline path. Not applied in the default
    /// `mix_or_process` path (requires pipeline mode).
    width_config: Option<StereoWidthConfig>,
    /// Optional convolution reverb configuration. Stored for use with the
    /// ChorusMasterPipeline path. Not applied in the default `mix_or_process`
    /// path (requires pipeline mode).
    convolution_config: Option<ConvolutionConfig>,
    /// Optional per-voice pitch correction (auto-tune / scale snapping).
    /// When `Some`, pitch correction is applied per-voice after vibrato and
    /// before transient shaping in the default mixing path.
    pitch_correct_config: Option<PitchCorrectConfig>,
    /// Optional per-voice harmonic exciter (presence/air enhancement).
    /// When `Some`, the exciter is applied per-voice after EQ in the default
    /// mixing path.
    exciter_config: Option<ExciterConfig>,
    /// Optional per-voice ADT vocal doubler.
    /// When `Some`, automatic double tracking is applied per-voice after
    /// humanize in the default mixing path.
    doubler_config: Option<DoublerConfig>,
    /// Optional per-voice spectral ducking (lead voice prominence).
    /// When `Some`, ducking is applied per-voice after bleed in the default
    /// mixing path.
    ducking_config: Option<DuckingConfig>,
    /// Persistent spectral ducker state (envelope followers per voice).
    /// Created lazily from `ducking_config`.
    ducker: Option<SpectralDucker>,
    /// Optional bus gain staging (LUFS targeting + peak normalization).
    /// Stored for use with the ChorusMasterPipeline path. Not applied in
    /// the default `mix_or_process` path (pipeline mode only).
    gain_staging_config: Option<GainStagingConfig>,
    /// Optional bus dithering (TPDF + noise shaping).
    /// Stored for use with the ChorusMasterPipeline path. Not applied in
    /// the default `mix_or_process` path (pipeline mode only).
    dither_config: Option<DitherConfig>,
    /// Optional per-voice noise gate configuration. When `Some`, noise gating
    /// is applied per-voice FIRST in the pipeline. Stored for use with the
    /// ChorusMasterPipeline path.
    gate_config: Option<GateConfig>,
    /// Optional per-voice timbral character variation configuration. When `Some`,
    /// character variation is applied per-voice after alignment, before vibrato.
    /// Stored for use with the ChorusMasterPipeline path.
    character_config: Option<CharacterConfig>,
    /// Optional bus early reflections room configuration. When `Some`,
    /// image-source early reflections are applied before Schroeder reverb.
    /// Stored for use with the ChorusMasterPipeline path.
    room_config: Option<RoomConfig>,
    /// Optional bus multi-band stereo configuration. When `Some`,
    /// frequency-dependent stereo width is applied after stereo width,
    /// before dynamics. Stored for use with the ChorusMasterPipeline path.
    multiband_stereo_config: Option<MultibandStereoConfig>,
    /// Optional bus spectral freeze configuration. When `Some`, spectral
    /// freeze is applied after reverb/convolution, before gain staging.
    /// Stored for use with the ChorusMasterPipeline path.
    freeze_config: Option<FreezeConfig>,
    /// Optional HRTF binaural spatial configuration. When `Some`, HRTF
    /// replaces the default stereo/spatial mix with binaural processing.
    /// Stored for use with the ChorusMasterPipeline path.
    hrtf_config: Option<HrtfConfig>,
    /// Optional per-voice auto-EQ configuration. When `Some`, spectral
    /// correction is applied per-voice after EQ/de-essing, before exciter.
    /// Stored for use with the ChorusMasterPipeline path.
    auto_eq_config: Option<AutoEqConfig>,
    /// Optional bus loudness normalization configuration. When `Some`,
    /// LUFS metering and normalization is applied after gain staging,
    /// before the limiter. Stored for use with the ChorusMasterPipeline path.
    loudness_config: Option<LoudnessConfig>,
    /// Optional per-voice sibilance processing configuration. When `Some`,
    /// frequency-domain de-essing is applied per-voice after auto-EQ,
    /// before humanize. Stored for use with the ChorusMasterPipeline path.
    sibilance_config: Option<SibilanceConfig>,
    /// Optional bus ensemble processor configuration. When `Some`, stereo
    /// modulation and diffusion is applied after convolution/freeze, before
    /// gain staging. Stored for use with the ChorusMasterPipeline path.
    ensemble_config: Option<EnsembleConfig>,
    /// Optional per-voice warmth (tube saturation + low-shelf enhancement).
    /// When `Some`, warmth is applied per-voice after exciter, before humanize
    /// in the ChorusMasterPipeline path.
    warmth_config: Option<WarmthConfig>,
    /// Optional bus stereo analysis (phase coherence monitoring + correction).
    /// When `Some`, stereo analysis is applied after width/multiband, before
    /// dynamics in the ChorusMasterPipeline path.
    stereo_analysis_config: Option<StereoAnalysisConfig>,
    /// Optional per-voice formant tuning (formant shift without pitch change).
    /// When `Some`, formant tuning is applied per-voice after character, before
    /// vibrato in the ChorusMasterPipeline path.
    formant_tune_config: Option<FormantTuneConfig>,
    /// Optional per-voice micro-pitch variation (slow random drift).
    /// When `Some`, micro-pitch is applied per-voice after detuning, before
    /// EQ in the ChorusMasterPipeline path.
    micro_pitch_config: Option<MicroPitchConfig>,
    /// Optional mix automation controller (scene transitions + gain automation).
    /// API-only: not wired into process(). Callers use this to drive real-time
    /// scene transitions externally.
    mix_automator: Option<MixAutomator>,
    /// Optional per-voice intonation tracking (pitch drift correction).
    /// When `Some`, intonation tracking is applied per-voice after vibrato,
    /// before pitch correction in the ChorusMasterPipeline path.
    intonation_config: Option<IntonationConfig>,
    /// Optional crossfade optimizer for streaming path. When `Some`,
    /// mixed audio chunks are processed through adaptive crossfade
    /// boundary optimization before assembly. Streaming path only.
    pub(crate) crossfade_optimizer: Option<CrossfadeOptimizer>,
    /// Optional per-voice spectral envelope matching configuration. When `Some`,
    /// spectral matching is applied per-voice after intonation, before detuning
    /// in the ChorusMasterPipeline path.
    spectral_match_config: Option<SpectralMatchConfig>,
    /// Optional bus sub-bass enhancement configuration. When `Some`, sub-bass
    /// is applied on the bus after dynamics, before saturation in the
    /// ChorusMasterPipeline path.
    sub_bass_config: Option<SubBassConfig>,
    /// Optional bus adaptive dynamics configuration. When `Some`, adaptive
    /// dynamics replaces/supplements regular dynamics in the ChorusMasterPipeline path.
    adaptive_dynamics_config: Option<AdaptiveDynamicsConfig>,
    /// Optional bus spectral tilt configuration. When `Some`, tilt is applied
    /// on the bus after gain staging, before limiter in the ChorusMasterPipeline path.
    tilt_config: Option<TiltConfig>,
    /// Optional per-voice onset synchronization configuration. When `Some`,
    /// onset sync is applied per-voice after alignment, before formant_tune
    /// in the ChorusMasterPipeline path.
    onset_sync_config: Option<OnsetSyncConfig>,
    /// Optional oversampling configuration. When `Some`, wraps saturation
    /// with 2x/4x oversampling in the ChorusMasterPipeline path.
    oversample_config: Option<OversampleConfig>,
}

impl KokoroChorus {
    /// Create a chorus pool from a primary `CompiledKokoro`.
    ///
    /// Creates `config.n_voices` instances via `clone_dispatch()`, each
    /// sharing GPU weight buffers with the primary. The primary instance
    /// is borrowed — the caller retains it and can still use it.
    ///
    /// Call `synthesize()` on the primary instance first to populate segment
    /// caches before creating the chorus. This ensures cloned instances inherit
    /// compiled segments via `SegmentCache::with_shared_weights()`.
    ///
    /// # Arguments
    ///
    /// * `primary` - A compiled Kokoro instance (ideally warmed up).
    /// * `config` - Chorus configuration (voice count, gains).
    pub fn new(
        primary: &CompiledKokoro,
        config: ChorusConfig,
    ) -> Result<Self, CompiledKokoroError> {
        config.validate()?;
        let voices: Vec<CompiledKokoro> = (0..config.n_voices)
            .map(|_| primary.clone_dispatch_warm())
            .collect();
        Ok(Self {
            voices,
            config,
            dynamics_preset: None,
            compressor: None,
            limiter: None,
            alignment_config: None,
            humanize_config: None,
            detune_config: None,
            saturation_config: None,
            saturation_processor: None,
            chorus_pipeline: None,
            stereo_config: None,
            mix_bus: None,
            mix_bus_config: None,
            formant_config: None,
            breath_config: None,
            breath_generator: None,
            spatial_config: None,
            transient_config: None,
            bleed_config: None,
            width_config: None,
            convolution_config: None,
            pitch_correct_config: None,
            exciter_config: None,
            doubler_config: None,
            ducking_config: None,
            ducker: None,
            gain_staging_config: None,
            dither_config: None,
            gate_config: None,
            character_config: None,
            room_config: None,
            multiband_stereo_config: None,
            freeze_config: None,
            hrtf_config: None,
            auto_eq_config: None,
            loudness_config: None,
            sibilance_config: None,
            ensemble_config: None,
            warmth_config: None,
            stereo_analysis_config: None,
            formant_tune_config: None,
            micro_pitch_config: None,
            mix_automator: None,
            intonation_config: None,
            crossfade_optimizer: None,
            spectral_match_config: None,
            sub_bass_config: None,
            adaptive_dynamics_config: None,
            tilt_config: None,
            onset_sync_config: None,
            oversample_config: None,
        })
    }

    /// Create a chorus pool from a warmed-up primary, sharing compiled segments.
    ///
    /// **Deprecated:** [`new()`](Self::new) now uses warm cloning (`clone_dispatch_warm`)
    /// and is functionally identical. Use `new()` instead.
    ///
    /// Part of #4104, #4305.
    #[deprecated(
        since = "0.1.0",
        note = "use KokoroChorus::new() which now uses warm cloning"
    )]
    pub fn new_warm(
        primary: &CompiledKokoro,
        config: ChorusConfig,
    ) -> Result<Self, CompiledKokoroError> {
        config.validate()?;
        let voices: Vec<CompiledKokoro> = (0..config.n_voices)
            .map(|_| primary.clone_dispatch_warm())
            .collect();
        Ok(Self {
            voices,
            config,
            dynamics_preset: None,
            compressor: None,
            limiter: None,
            alignment_config: None,
            humanize_config: None,
            detune_config: None,
            saturation_config: None,
            saturation_processor: None,
            chorus_pipeline: None,
            stereo_config: None,
            mix_bus: None,
            mix_bus_config: None,
            formant_config: None,
            breath_config: None,
            breath_generator: None,
            spatial_config: None,
            transient_config: None,
            bleed_config: None,
            width_config: None,
            convolution_config: None,
            pitch_correct_config: None,
            exciter_config: None,
            doubler_config: None,
            ducking_config: None,
            ducker: None,
            gain_staging_config: None,
            dither_config: None,
            gate_config: None,
            character_config: None,
            room_config: None,
            multiband_stereo_config: None,
            freeze_config: None,
            hrtf_config: None,
            auto_eq_config: None,
            loudness_config: None,
            sibilance_config: None,
            ensemble_config: None,
            warmth_config: None,
            stereo_analysis_config: None,
            formant_tune_config: None,
            micro_pitch_config: None,
            mix_automator: None,
            intonation_config: None,
            crossfade_optimizer: None,
            spectral_match_config: None,
            sub_bass_config: None,
            adaptive_dynamics_config: None,
            tilt_config: None,
            onset_sync_config: None,
            oversample_config: None,
        })
    }

    /// Create a chorus pool from a [`SharedSegmentCache`].
    ///
    /// Each voice instance shares compiled Metal pipelines and GPU weight
    /// buffers via `Arc<CompiledModelDef>`, eliminating per-voice recompilation.
    /// The shared cache should be created from a warmed-up primary instance
    /// (after at least one `synthesize()` call).
    ///
    /// This is more efficient than [`new()`](Self::new) for large voice counts
    /// because each instance gets pre-populated segment caches with shared
    /// compiled model definitions, not just shared weight aliases.
    ///
    /// # Arguments
    ///
    /// * `shared_cache` - Pre-warmed shared segment cache.
    /// * `config` - Chorus configuration (voice count, gains).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_metal::compiled_kokoro::{CompiledKokoro, SharedSegmentCache, KokoroChorus};
    /// use nn_models::kokoro_chorus::ChorusConfig;
    ///
    /// let mut primary = CompiledKokoro::new(model)?;
    /// let _ = primary.synthesize(&input_ids, &style, 1.0, &cache)?;
    /// let shared = SharedSegmentCache::from_compiled(&primary);
    /// let chorus = KokoroChorus::from_shared_cache(&shared, ChorusConfig::equal_gain(8)?)?;
    /// ```
    ///
    /// Part of #4104.
    pub fn from_shared_cache(
        shared_cache: &super::SharedSegmentCache,
        config: ChorusConfig,
    ) -> Result<Self, CompiledKokoroError> {
        config.validate()?;
        let voices: Vec<CompiledKokoro> = (0..config.n_voices)
            .map(|_| shared_cache.create_instance())
            .collect();
        Ok(Self {
            voices,
            config,
            dynamics_preset: None,
            compressor: None,
            limiter: None,
            alignment_config: None,
            humanize_config: None,
            detune_config: None,
            saturation_config: None,
            saturation_processor: None,
            chorus_pipeline: None,
            stereo_config: None,
            mix_bus: None,
            mix_bus_config: None,
            formant_config: None,
            breath_config: None,
            breath_generator: None,
            spatial_config: None,
            transient_config: None,
            bleed_config: None,
            width_config: None,
            convolution_config: None,
            pitch_correct_config: None,
            exciter_config: None,
            doubler_config: None,
            ducking_config: None,
            ducker: None,
            gain_staging_config: None,
            dither_config: None,
            gate_config: None,
            character_config: None,
            room_config: None,
            multiband_stereo_config: None,
            freeze_config: None,
            hrtf_config: None,
            auto_eq_config: None,
            loudness_config: None,
            sibilance_config: None,
            ensemble_config: None,
            warmth_config: None,
            stereo_analysis_config: None,
            formant_tune_config: None,
            micro_pitch_config: None,
            mix_automator: None,
            intonation_config: None,
            crossfade_optimizer: None,
            spectral_match_config: None,
            sub_bass_config: None,
            adaptive_dynamics_config: None,
            tilt_config: None,
            onset_sync_config: None,
            oversample_config: None,
        })
    }

    /// Number of voices in the chorus.
    #[must_use]
    pub fn n_voices(&self) -> usize {
        self.voices.len()
    }

    /// Access the chorus configuration.
    #[must_use]
    pub fn config(&self) -> &ChorusConfig {
        &self.config
    }

    /// Enable per-voice humanization (breathing, micro-timing, amplitude envelope).
    ///
    /// When set, each voice's PCM audio is independently humanized before
    /// mixing. The `voice_index` seeds deterministic per-voice variation,
    /// so each voice gets different breathing patterns, onset jitter, and
    /// amplitude envelopes.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_models::kokoro_chorus_humanize::HumanizeConfig;
    ///
    /// let chorus = KokoroChorus::new(&primary, config)?
    ///     .with_humanize(HumanizeConfig::default());
    /// ```
    #[must_use]
    pub fn with_humanize(mut self, config: HumanizeConfig) -> Self {
        self.humanize_config = Some(config);
        self
    }

    /// Access the humanization config, if set.
    #[must_use]
    pub fn humanize_config(&self) -> Option<&HumanizeConfig> {
        self.humanize_config.as_ref()
    }

    /// Enable per-voice breath noise at detected pauses.
    ///
    /// When set, synthetic breath sounds are inserted at pause regions with
    /// per-voice timing stagger. Applied after humanization and before mixing.
    /// Each voice gets different breath timing via a deterministic PRNG offset.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails
    /// validation.
    pub fn with_breath(mut self, config: BreathConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("breath config: {e}")))?;
        let n = self.voices.len();
        let generator = BreathGenerator::new(&config, n)
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("breath generator: {e}")))?;
        self.breath_config = Some(config);
        self.breath_generator = Some(generator);
        Ok(self)
    }

    /// Access the breath config, if set.
    #[must_use]
    pub fn breath_config(&self) -> Option<&BreathConfig> {
        self.breath_config.as_ref()
    }

    /// Whether breath insertion is enabled.
    #[must_use]
    pub fn has_breath(&self) -> bool {
        self.breath_config.is_some()
    }

    /// Enable per-voice spatial depth processing.
    ///
    /// When set and no chorus pipeline is active, each voice is processed
    /// through distance-based attenuation, air absorption lowpass,
    /// propagation delay, and ILD stereo panning before mixing.
    #[must_use]
    pub fn with_spatial(mut self, config: SpatialConfig) -> Self {
        self.spatial_config = Some(config);
        self
    }

    /// Whether spatial depth processing is enabled.
    #[must_use]
    pub fn has_spatial(&self) -> bool {
        self.spatial_config.is_some()
    }

    /// Access the spatial config, if set.
    #[must_use]
    pub fn spatial_config(&self) -> Option<&SpatialConfig> {
        self.spatial_config.as_ref()
    }

    /// Enable per-voice transient shaping (attack/sustain control).
    ///
    /// When set, transient shaping is applied after alignment, before detune
    /// in the default mixing path. Shapes consonant attacks and vowel sustain.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails validation.
    pub fn with_transient(mut self, config: TransientConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("transient config: {e}")))?;
        self.transient_config = Some(config);
        Ok(self)
    }

    /// Access the transient config, if set.
    #[must_use]
    pub fn transient_config(&self) -> Option<&TransientConfig> {
        self.transient_config.as_ref()
    }

    /// Whether transient shaping is enabled.
    #[must_use]
    pub fn has_transient(&self) -> bool {
        self.transient_config.is_some()
    }

    /// Enable per-voice bleed (microphone crosstalk simulation).
    ///
    /// When set, voice bleed is applied after breath, before spatial/mix
    /// in the default mixing path.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails validation.
    pub fn with_bleed(mut self, config: BleedConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("bleed config: {e}")))?;
        self.bleed_config = Some(config);
        Ok(self)
    }

    /// Access the bleed config, if set.
    #[must_use]
    pub fn bleed_config(&self) -> Option<&BleedConfig> {
        self.bleed_config.as_ref()
    }

    /// Whether voice bleed is enabled.
    #[must_use]
    pub fn has_bleed(&self) -> bool {
        self.bleed_config.is_some()
    }

    /// Enable stereo width enhancement configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Width processing
    /// requires pipeline mode (stereo L/R buses) and is not applied in the
    /// default `mix_or_process` path.
    #[must_use]
    pub fn with_width_config(mut self, config: StereoWidthConfig) -> Self {
        self.width_config = Some(config);
        self
    }

    /// Access the width config, if set.
    #[must_use]
    pub fn width_config(&self) -> Option<&StereoWidthConfig> {
        self.width_config.as_ref()
    }

    /// Whether stereo width enhancement is configured.
    #[must_use]
    pub fn has_width(&self) -> bool {
        self.width_config.is_some()
    }

    /// Enable convolution reverb configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Convolution reverb
    /// requires pipeline mode (stereo L/R buses) and is not applied in the
    /// default `mix_or_process` path.
    #[must_use]
    pub fn with_convolution_config(mut self, config: ConvolutionConfig) -> Self {
        self.convolution_config = Some(config);
        self
    }

    /// Access the convolution config, if set.
    #[must_use]
    pub fn convolution_config(&self) -> Option<&ConvolutionConfig> {
        self.convolution_config.as_ref()
    }

    /// Whether convolution reverb is configured.
    #[must_use]
    pub fn has_convolution(&self) -> bool {
        self.convolution_config.is_some()
    }

    /// Enable per-voice pitch correction (auto-tune / scale snapping).
    ///
    /// When set, pitch correction is applied per-voice after alignment/vibrato
    /// and before transient shaping in the default mixing path.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails validation.
    pub fn with_pitch_correct(
        mut self,
        config: PitchCorrectConfig,
    ) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("pitch_correct config: {e}")))?;
        self.pitch_correct_config = Some(config);
        Ok(self)
    }

    /// Access the pitch correction config, if set.
    #[must_use]
    pub fn pitch_correct_config(&self) -> Option<&PitchCorrectConfig> {
        self.pitch_correct_config.as_ref()
    }

    /// Whether pitch correction is enabled.
    #[must_use]
    pub fn has_pitch_correct(&self) -> bool {
        self.pitch_correct_config.is_some()
    }

    /// Enable per-voice harmonic exciter (presence/air enhancement).
    ///
    /// When set, the exciter is applied per-voice after EQ to add harmonics
    /// and air-band shimmer in the default mixing path.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails validation.
    pub fn with_exciter(mut self, config: ExciterConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("exciter config: {e}")))?;
        self.exciter_config = Some(config);
        Ok(self)
    }

    /// Access the exciter config, if set.
    #[must_use]
    pub fn exciter_config(&self) -> Option<&ExciterConfig> {
        self.exciter_config.as_ref()
    }

    /// Whether the exciter is enabled.
    #[must_use]
    pub fn has_exciter(&self) -> bool {
        self.exciter_config.is_some()
    }

    /// Enable per-voice ADT vocal doubler.
    ///
    /// When set, automatic double tracking is applied per-voice after humanize
    /// in the default mixing path, creating doubled copies with timing/pitch
    /// variation.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails validation.
    pub fn with_doubler(mut self, config: DoublerConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("doubler config: {e}")))?;
        self.doubler_config = Some(config);
        Ok(self)
    }

    /// Access the doubler config, if set.
    #[must_use]
    pub fn doubler_config(&self) -> Option<&DoublerConfig> {
        self.doubler_config.as_ref()
    }

    /// Whether the doubler is enabled.
    #[must_use]
    pub fn has_doubler(&self) -> bool {
        self.doubler_config.is_some()
    }

    /// Enable per-voice spectral ducking (lead voice prominence).
    ///
    /// When set, ducking is applied per-voice after bleed in the default
    /// mixing path to reduce non-lead voices when the lead is active.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails validation.
    pub fn with_ducking(mut self, config: DuckingConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("ducking config: {e}")))?;
        let ducker = SpectralDucker::new(&config, KOKORO_SAMPLE_RATE as f32)
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("ducking: {e}")))?;
        self.ducking_config = Some(config);
        self.ducker = Some(ducker);
        Ok(self)
    }

    /// Access the ducking config, if set.
    #[must_use]
    pub fn ducking_config(&self) -> Option<&DuckingConfig> {
        self.ducking_config.as_ref()
    }

    /// Whether ducking is enabled.
    #[must_use]
    pub fn has_ducking(&self) -> bool {
        self.ducking_config.is_some()
    }

    /// Enable bus gain staging configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Gain staging is
    /// not applied in the default `mix_or_process` path (pipeline mode only).
    #[must_use]
    pub fn with_gain_staging_config(mut self, config: GainStagingConfig) -> Self {
        self.gain_staging_config = Some(config);
        self
    }

    /// Access the gain staging config, if set.
    #[must_use]
    pub fn gain_staging_config(&self) -> Option<&GainStagingConfig> {
        self.gain_staging_config.as_ref()
    }

    /// Whether gain staging is configured.
    #[must_use]
    pub fn has_gain_staging(&self) -> bool {
        self.gain_staging_config.is_some()
    }

    /// Enable bus dithering configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Dithering is
    /// not applied in the default `mix_or_process` path (pipeline mode only).
    #[must_use]
    pub fn with_dither_config(mut self, config: DitherConfig) -> Self {
        self.dither_config = Some(config);
        self
    }

    /// Access the dither config, if set.
    #[must_use]
    pub fn dither_config(&self) -> Option<&DitherConfig> {
        self.dither_config.as_ref()
    }

    /// Whether dithering is configured.
    #[must_use]
    pub fn has_dither(&self) -> bool {
        self.dither_config.is_some()
    }

    /// Enable per-voice noise gate configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Noise gating is
    /// applied per-voice FIRST in the pipeline, before alignment.
    #[must_use]
    pub fn with_gate(mut self, config: GateConfig) -> Self {
        self.gate_config = Some(config);
        self
    }

    /// Access the gate config, if set.
    #[must_use]
    pub fn gate_config(&self) -> Option<&GateConfig> {
        self.gate_config.as_ref()
    }

    /// Whether noise gate is configured.
    #[must_use]
    pub fn has_gate(&self) -> bool {
        self.gate_config.is_some()
    }

    /// Enable per-voice timbral character variation configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Character variation
    /// is applied per-voice after alignment, before vibrato.
    #[must_use]
    pub fn with_character(mut self, config: CharacterConfig) -> Self {
        self.character_config = Some(config);
        self
    }

    /// Access the character config, if set.
    #[must_use]
    pub fn character_config(&self) -> Option<&CharacterConfig> {
        self.character_config.as_ref()
    }

    /// Whether character variation is configured.
    #[must_use]
    pub fn has_character(&self) -> bool {
        self.character_config.is_some()
    }

    /// Enable bus early reflections room simulation configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Early reflections
    /// are applied before Schroeder reverb.
    #[must_use]
    pub fn with_room(mut self, config: RoomConfig) -> Self {
        self.room_config = Some(config);
        self
    }

    /// Access the room config, if set.
    #[must_use]
    pub fn room_config(&self) -> Option<&RoomConfig> {
        self.room_config.as_ref()
    }

    /// Whether room early reflections are configured.
    #[must_use]
    pub fn has_room(&self) -> bool {
        self.room_config.is_some()
    }

    /// Enable bus multi-band stereo configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Multi-band stereo
    /// is applied after stereo width, before dynamics.
    #[must_use]
    pub fn with_multiband_stereo(mut self, config: MultibandStereoConfig) -> Self {
        self.multiband_stereo_config = Some(config);
        self
    }

    /// Access the multiband stereo config, if set.
    #[must_use]
    pub fn multiband_stereo_config(&self) -> Option<&MultibandStereoConfig> {
        self.multiband_stereo_config.as_ref()
    }

    /// Whether multi-band stereo is configured.
    #[must_use]
    pub fn has_multiband_stereo(&self) -> bool {
        self.multiband_stereo_config.is_some()
    }

    /// Enable bus spectral freeze configuration.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Spectral freeze
    /// is applied after reverb/convolution, before gain staging.
    #[must_use]
    pub fn with_freeze(mut self, config: FreezeConfig) -> Self {
        self.freeze_config = Some(config);
        self
    }

    /// Access the freeze config, if set.
    #[must_use]
    pub fn freeze_config(&self) -> Option<&FreezeConfig> {
        self.freeze_config.as_ref()
    }

    /// Whether spectral freeze is configured.
    #[must_use]
    pub fn has_freeze(&self) -> bool {
        self.freeze_config.is_some()
    }

    /// Enable HRTF binaural spatial processing.
    ///
    /// Stored for use with the ChorusMasterPipeline path. When set, HRTF
    /// replaces the default stereo/spatial mix with binaural processing
    /// (ITD + ILD + head shadow) for immersive output.
    #[must_use]
    pub fn with_hrtf(mut self, config: HrtfConfig) -> Self {
        self.hrtf_config = Some(config);
        self
    }

    /// Access the HRTF config, if set.
    #[must_use]
    pub fn hrtf_config(&self) -> Option<&HrtfConfig> {
        self.hrtf_config.as_ref()
    }

    /// Whether HRTF is configured.
    #[must_use]
    pub fn has_hrtf(&self) -> bool {
        self.hrtf_config.is_some()
    }

    /// Enable per-voice auto-EQ (spectral correction).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after EQ/de-essing, before exciter.
    #[must_use]
    pub fn with_auto_eq(mut self, config: AutoEqConfig) -> Self {
        self.auto_eq_config = Some(config);
        self
    }

    /// Access the auto-EQ config, if set.
    #[must_use]
    pub fn auto_eq_config(&self) -> Option<&AutoEqConfig> {
        self.auto_eq_config.as_ref()
    }

    /// Whether auto-EQ is configured.
    #[must_use]
    pub fn has_auto_eq(&self) -> bool {
        self.auto_eq_config.is_some()
    }

    /// Enable bus loudness normalization (LUFS metering).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied on the
    /// bus after gain staging, before the limiter.
    #[must_use]
    pub fn with_loudness(mut self, config: LoudnessConfig) -> Self {
        self.loudness_config = Some(config);
        self
    }

    /// Access the loudness config, if set.
    #[must_use]
    pub fn loudness_config(&self) -> Option<&LoudnessConfig> {
        self.loudness_config.as_ref()
    }

    /// Whether loudness normalization is configured.
    #[must_use]
    pub fn has_loudness(&self) -> bool {
        self.loudness_config.is_some()
    }

    /// Enable per-voice sibilance processing (frequency-domain de-essing).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after auto-EQ, before humanize.
    #[must_use]
    pub fn with_sibilance(mut self, config: SibilanceConfig) -> Self {
        self.sibilance_config = Some(config);
        self
    }

    /// Access the sibilance config, if set.
    #[must_use]
    pub fn sibilance_config(&self) -> Option<&SibilanceConfig> {
        self.sibilance_config.as_ref()
    }

    /// Whether sibilance processing is configured.
    #[must_use]
    pub fn has_sibilance(&self) -> bool {
        self.sibilance_config.is_some()
    }

    /// Enable bus ensemble processor (stereo modulation/diffusion).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied on the
    /// bus after convolution/freeze, before gain staging.
    #[must_use]
    pub fn with_ensemble(mut self, config: EnsembleConfig) -> Self {
        self.ensemble_config = Some(config);
        self
    }

    /// Access the ensemble config, if set.
    #[must_use]
    pub fn ensemble_config(&self) -> Option<&EnsembleConfig> {
        self.ensemble_config.as_ref()
    }

    /// Whether ensemble processing is configured.
    #[must_use]
    pub fn has_ensemble(&self) -> bool {
        self.ensemble_config.is_some()
    }

    /// Enable per-voice warmth processing (tube saturation + low-shelf).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after exciter, before humanize.
    #[must_use]
    pub fn with_warmth(mut self, config: WarmthConfig) -> Self {
        self.warmth_config = Some(config);
        self
    }

    /// Access the warmth config, if set.
    #[must_use]
    pub fn warmth_config(&self) -> Option<&WarmthConfig> {
        self.warmth_config.as_ref()
    }

    /// Whether warmth processing is configured.
    #[must_use]
    pub fn has_warmth(&self) -> bool {
        self.warmth_config.is_some()
    }

    /// Enable bus stereo analysis (phase coherence monitoring + correction).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied on the bus
    /// after width/multiband stereo, before dynamics.
    #[must_use]
    pub fn with_stereo_analysis(mut self, config: StereoAnalysisConfig) -> Self {
        self.stereo_analysis_config = Some(config);
        self
    }

    /// Access the stereo analysis config, if set.
    #[must_use]
    pub fn stereo_analysis_config(&self) -> Option<&StereoAnalysisConfig> {
        self.stereo_analysis_config.as_ref()
    }

    /// Whether stereo analysis is configured.
    #[must_use]
    pub fn has_stereo_analysis(&self) -> bool {
        self.stereo_analysis_config.is_some()
    }

    /// Enable per-voice formant tuning (formant shift without pitch change).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after character, before vibrato.
    #[must_use]
    pub fn with_formant_tune(mut self, config: FormantTuneConfig) -> Self {
        self.formant_tune_config = Some(config);
        self
    }

    /// Access the formant tune config, if set.
    #[must_use]
    pub fn formant_tune_config(&self) -> Option<&FormantTuneConfig> {
        self.formant_tune_config.as_ref()
    }

    /// Whether formant tuning is configured.
    #[must_use]
    pub fn has_formant_tune(&self) -> bool {
        self.formant_tune_config.is_some()
    }

    /// Enable per-voice micro-pitch variation (slow random drift).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after detuning, before EQ.
    #[must_use]
    pub fn with_micro_pitch(mut self, config: MicroPitchConfig) -> Self {
        self.micro_pitch_config = Some(config);
        self
    }

    /// Access the micro-pitch config, if set.
    #[must_use]
    pub fn micro_pitch_config(&self) -> Option<&MicroPitchConfig> {
        self.micro_pitch_config.as_ref()
    }

    /// Whether micro-pitch variation is configured.
    #[must_use]
    pub fn has_micro_pitch(&self) -> bool {
        self.micro_pitch_config.is_some()
    }

    /// Enable mix automation (scene transitions + gain automation).
    ///
    /// API-only: the automator is not wired into `process()`. Callers use
    /// the returned accessor to drive real-time scene transitions externally.
    #[must_use]
    pub fn with_mix_automator(mut self, automator: MixAutomator) -> Self {
        self.mix_automator = Some(automator);
        self
    }

    /// Access the mix automator, if set.
    #[must_use]
    pub fn mix_automator(&self) -> Option<&MixAutomator> {
        self.mix_automator.as_ref()
    }

    /// Mutable access to the mix automator for scene changes.
    pub fn mix_automator_mut(&mut self) -> Option<&mut MixAutomator> {
        self.mix_automator.as_mut()
    }

    /// Whether mix automation is configured.
    #[must_use]
    pub fn has_mix_automator(&self) -> bool {
        self.mix_automator.is_some()
    }

    /// Enable per-voice intonation tracking (pitch drift correction).
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after vibrato, before pitch correction.
    #[must_use]
    pub fn with_intonation(mut self, config: IntonationConfig) -> Self {
        self.intonation_config = Some(config);
        self
    }

    /// Access the intonation config, if set.
    #[must_use]
    pub fn intonation_config(&self) -> Option<&IntonationConfig> {
        self.intonation_config.as_ref()
    }

    /// Whether intonation tracking is configured.
    #[must_use]
    pub fn has_intonation(&self) -> bool {
        self.intonation_config.is_some()
    }

    /// Enable adaptive crossfade optimization for the streaming path.
    ///
    /// When set, mixed audio chunks are processed through adaptive crossfade
    /// boundary detection (zero-crossing, energy analysis) before assembly.
    /// This only affects the streaming synthesis methods.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails
    /// validation.
    pub fn with_crossfade_optimizer(
        mut self,
        config: CrossfadeOptimizerConfig,
    ) -> Result<Self, CompiledKokoroError> {
        config.validate().map_err(|e| {
            CompiledKokoroError::InvalidInput(format!("crossfade optimizer config: {e}"))
        })?;
        let optimizer = CrossfadeOptimizer::new(config)
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("crossfade optimizer: {e}")))?;
        self.crossfade_optimizer = Some(optimizer);
        Ok(self)
    }

    /// Access the crossfade optimizer, if set.
    #[must_use]
    pub fn crossfade_optimizer(&self) -> Option<&CrossfadeOptimizer> {
        self.crossfade_optimizer.as_ref()
    }

    /// Whether crossfade optimization is enabled.
    #[must_use]
    pub fn has_crossfade_optimizer(&self) -> bool {
        self.crossfade_optimizer.is_some()
    }

    /// Enable per-voice spectral envelope matching.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after intonation, before detuning.
    #[must_use]
    pub fn with_spectral_match(mut self, config: SpectralMatchConfig) -> Self {
        self.spectral_match_config = Some(config);
        self
    }

    /// Access the spectral match config, if set.
    #[must_use]
    pub fn spectral_match_config(&self) -> Option<&SpectralMatchConfig> {
        self.spectral_match_config.as_ref()
    }

    /// Whether spectral matching is configured.
    #[must_use]
    pub fn has_spectral_match(&self) -> bool {
        self.spectral_match_config.is_some()
    }

    /// Enable bus sub-harmonic bass enhancement.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied on the
    /// bus after dynamics, before saturation.
    #[must_use]
    pub fn with_sub_bass(mut self, config: SubBassConfig) -> Self {
        self.sub_bass_config = Some(config);
        self
    }

    /// Access the sub-bass config, if set.
    #[must_use]
    pub fn sub_bass_config(&self) -> Option<&SubBassConfig> {
        self.sub_bass_config.as_ref()
    }

    /// Whether sub-bass enhancement is configured.
    #[must_use]
    pub fn has_sub_bass(&self) -> bool {
        self.sub_bass_config.is_some()
    }

    /// Enable bus adaptive dynamics with masking.
    ///
    /// Stored for use with the ChorusMasterPipeline path. When set alongside
    /// regular dynamics, adaptive dynamics replaces the regular compressor.
    #[must_use]
    pub fn with_adaptive_dynamics(mut self, config: AdaptiveDynamicsConfig) -> Self {
        self.adaptive_dynamics_config = Some(config);
        self
    }

    /// Access the adaptive dynamics config, if set.
    #[must_use]
    pub fn adaptive_dynamics_config(&self) -> Option<&AdaptiveDynamicsConfig> {
        self.adaptive_dynamics_config.as_ref()
    }

    /// Whether adaptive dynamics is configured.
    #[must_use]
    pub fn has_adaptive_dynamics(&self) -> bool {
        self.adaptive_dynamics_config.is_some()
    }

    /// Enable bus spectral tilt.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied on the
    /// bus after gain staging, before the limiter.
    #[must_use]
    pub fn with_tilt(mut self, config: TiltConfig) -> Self {
        self.tilt_config = Some(config);
        self
    }

    /// Access the tilt config, if set.
    #[must_use]
    pub fn tilt_config(&self) -> Option<&TiltConfig> {
        self.tilt_config.as_ref()
    }

    /// Whether spectral tilt is configured.
    #[must_use]
    pub fn has_tilt(&self) -> bool {
        self.tilt_config.is_some()
    }

    /// Enable per-voice onset synchronization.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Applied per-voice
    /// after alignment, before formant_tune.
    #[must_use]
    pub fn with_onset_sync(mut self, config: OnsetSyncConfig) -> Self {
        self.onset_sync_config = Some(config);
        self
    }

    /// Access the onset sync config, if set.
    #[must_use]
    pub fn onset_sync_config(&self) -> Option<&OnsetSyncConfig> {
        self.onset_sync_config.as_ref()
    }

    /// Whether onset synchronization is configured.
    #[must_use]
    pub fn has_onset_sync(&self) -> bool {
        self.onset_sync_config.is_some()
    }

    /// Enable oversampling for saturation/exciter stages.
    ///
    /// Stored for use with the ChorusMasterPipeline path. Wraps saturation
    /// with 2x/4x oversampling for anti-aliased waveshaping.
    #[must_use]
    pub fn with_oversample(mut self, config: OversampleConfig) -> Self {
        self.oversample_config = Some(config);
        self
    }

    /// Access the oversample config, if set.
    #[must_use]
    pub fn oversample_config(&self) -> Option<&OversampleConfig> {
        self.oversample_config.as_ref()
    }

    /// Whether oversampling is configured.
    #[must_use]
    pub fn has_oversample(&self) -> bool {
        self.oversample_config.is_some()
    }

    /// Enable adaptive voice alignment (cross-correlation temporal sync).
    ///
    /// When set, voices are aligned to a common temporal reference before
    /// any other processing (detuning, humanization, EQ, mix). Alignment
    /// must be applied first so that timing corrections are not undone by
    /// subsequent pitch or spectral modifications.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails
    /// validation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_models::kokoro_chorus_alignment::AlignmentConfig;
    ///
    /// let chorus = KokoroChorus::new(&primary, config)?
    ///     .with_alignment(AlignmentConfig::new(0.6)?)?;
    /// ```
    pub fn with_alignment(mut self, config: AlignmentConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("alignment config: {e}")))?;
        self.alignment_config = Some(config);
        Ok(self)
    }

    /// Access the alignment config, if set.
    #[must_use]
    pub fn alignment_config(&self) -> Option<&AlignmentConfig> {
        self.alignment_config.as_ref()
    }

    /// Whether alignment is enabled.
    #[must_use]
    pub fn has_alignment(&self) -> bool {
        self.alignment_config.is_some()
    }

    /// Enable per-voice detuning for natural pitch variation.
    ///
    /// When set, each voice's PCM audio is resampled at a slightly different
    /// rate using allpass Thiran interpolation. Voice 0 is always the anchor
    /// (undetuned). Other voices spread symmetrically from `-cents_spread`
    /// to `+cents_spread`, creating natural beating frequencies for a warm,
    /// thick ensemble sound.
    ///
    /// Typical values: 5-15 cents for subtle ensemble width, 15-30 for
    /// a wide, dramatic chorus effect.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails
    /// validation (e.g., `cents_spread` out of [0.0, 50.0]).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_models::kokoro_chorus_detune::DetuneConfig;
    ///
    /// let chorus = KokoroChorus::new(&primary, config)?
    ///     .with_detune(DetuneConfig::default())?;
    /// ```
    pub fn with_detune(mut self, config: DetuneConfig) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("detune config: {e}")))?;
        self.detune_config = Some(config);
        Ok(self)
    }

    /// Access the detuning config, if set.
    #[must_use]
    pub fn detune_config(&self) -> Option<&DetuneConfig> {
        self.detune_config.as_ref()
    }

    /// Whether detuning is enabled.
    #[must_use]
    pub fn has_detune(&self) -> bool {
        self.detune_config.is_some()
    }

    /// Enable formant-preserving pitch shift for detuning.
    ///
    /// When set alongside `detune_config`, voices with large detuning (>10
    /// cents) use the PSOLA formant-preserving algorithm instead of basic
    /// allpass resampling. This prevents formant distortion ("chipmunk
    /// effect") on voices with significant pitch offsets while keeping the
    /// fast path for small detuning amounts.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the config fails
    /// validation.
    pub fn with_formant_preserve(
        mut self,
        config: FormantPreserveConfig,
    ) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("formant config: {e}")))?;
        self.formant_config = Some(config);
        Ok(self)
    }

    /// Access the formant preservation config, if set.
    #[must_use]
    pub fn formant_config(&self) -> Option<&FormantPreserveConfig> {
        self.formant_config.as_ref()
    }

    /// Whether formant preservation is enabled.
    #[must_use]
    pub fn has_formant_preserve(&self) -> bool {
        self.formant_config.is_some()
    }

    /// Enable stereo imaging for the chorus output.
    ///
    /// When set, voices are panned using constant-power sin/cos panning
    /// and the output is interleaved stereo `[L0, R0, L1, R1, ...]`.
    /// When not set, the output remains mono.
    ///
    /// Use [`StereoChorusConfig::auto_layout`] for automatic voice placement.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_models::kokoro_chorus_stereo::StereoChorusConfig;
    ///
    /// let chorus = KokoroChorus::new(&primary, config)?
    ///     .with_stereo_config(StereoChorusConfig::auto_layout(8)?);
    /// ```
    #[must_use]
    pub fn with_stereo_config(mut self, config: StereoChorusConfig) -> Self {
        self.stereo_config = Some(config);
        self
    }

    /// Enable per-voice EQ and de-essing via a mix bus processor.
    ///
    /// When set, each voice's PCM audio is processed through a 3-band
    /// parametric EQ and RMS de-esser before mixing. After mixing, an
    /// optional bus EQ shapes the combined output.
    ///
    /// # Errors
    ///
    /// Returns `CompiledKokoroError::InvalidInput` if the MixBusConfig
    /// fails validation (e.g., invalid EQ frequencies).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_models::kokoro_chorus_eq::{MixBusConfig, EqPreset};
    ///
    /// let chorus = KokoroChorus::new(&primary, config)?
    ///     .with_eq_config(MixBusConfig::from_preset(EqPreset::Warm))?;
    /// ```
    pub fn with_eq_config(mut self, config: MixBusConfig) -> Result<Self, CompiledKokoroError> {
        let n = self.voices.len();
        let processor = MixBusProcessor::new(n, &config)
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("eq config: {e}")))?;
        self.mix_bus = Some(processor);
        self.mix_bus_config = Some(config);
        Ok(self)
    }

    /// Access the stereo imaging config, if set.
    #[must_use]
    pub fn stereo_config(&self) -> Option<&StereoChorusConfig> {
        self.stereo_config.as_ref()
    }

    /// Whether stereo imaging is enabled.
    #[must_use]
    pub fn has_stereo(&self) -> bool {
        self.stereo_config.is_some()
    }

    /// Access the mix bus config, if set.
    #[must_use]
    pub fn mix_bus_config(&self) -> Option<&MixBusConfig> {
        self.mix_bus_config.as_ref()
    }

    /// Whether per-voice EQ is enabled.
    #[must_use]
    pub fn has_eq(&self) -> bool {
        self.mix_bus.is_some()
    }

    /// Reset the mix bus processor state (EQ filter history + de-esser envelopes).
    ///
    /// Call this when starting a new synthesis session to clear filter state.
    /// No-op when EQ is not enabled.
    pub fn reset_eq(&mut self) {
        if let Some(ref mut bus) = self.mix_bus {
            bus.reset();
        }
    }

    /// Access a specific voice instance (e.g., for precompilation).
    #[must_use]
    pub fn voice(&self, index: usize) -> Option<&CompiledKokoro> {
        self.voices.get(index)
    }

    /// Access a specific voice instance mutably.
    pub fn voice_mut(&mut self, index: usize) -> Option<&mut CompiledKokoro> {
        self.voices.get_mut(index)
    }

    /// Enable multi-band dynamics compression on the mixed chorus output.
    ///
    /// When set, a 3-band compressor (Linkwitz-Riley crossover at 300 Hz and
    /// 4 kHz) and a brick-wall bus limiter are applied to every synthesis
    /// result **after** voice mixing and **before** returning the final PCM.
    ///
    /// The compressor and limiter are stateful (envelope followers) and persist
    /// across synthesis calls, providing smooth dynamics tracking for streaming
    /// use cases.
    ///
    /// # Presets
    ///
    /// - [`DynamicsPreset::Broadcast`] -- moderate compression, -1 dB ceiling.
    ///   Recommended for general use.
    /// - [`DynamicsPreset::Gentle`] -- light compression for natural dynamics.
    /// - [`DynamicsPreset::Aggressive`] -- heavy compression for dense mixes
    ///   (8+ voices).
    /// - [`DynamicsPreset::Mastering`] -- transparent limiting with gentle
    ///   multiband control.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut chorus = KokoroChorus::new(&primary, config)?
    ///     .with_dynamics(DynamicsPreset::Broadcast)?;
    /// ```
    pub fn with_dynamics(mut self, preset: DynamicsPreset) -> Result<Self, CompiledKokoroError> {
        let comp_config = preset.to_config();
        self.compressor = Some(
            MultibandCompressor::new(&comp_config)
                .map_err(|e| CompiledKokoroError::InvalidInput(format!("dynamics config: {e}")))?,
        );
        // Broadcast preset uses -1 dB ceiling; others use the default -0.1 dB.
        let limiter = match preset {
            DynamicsPreset::Broadcast => BusLimiter::with_ceiling_db(-1.0),
            _ => BusLimiter::new(),
        };
        self.limiter = Some(limiter);
        self.dynamics_preset = Some(preset);
        Ok(self)
    }

    /// Enable bus saturation (harmonic warmth) on the mixed chorus output.
    ///
    /// When set, a saturation processor (2x oversampled waveshaper with
    /// decimation filter) is applied after dynamics processing in the default
    /// mixing path. The processor is stateful (decimation filter state) and
    /// persists across synthesis calls.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_models::kokoro_chorus_saturation::{SaturationConfig, SaturationMode};
    ///
    /// let chorus = KokoroChorus::new(&primary, config)?
    ///     .with_saturation(
    ///         SaturationConfig::new()
    ///             .with_drive(0.15)
    ///             .with_mode(SaturationMode::Warm),
    ///     )?;
    /// ```
    pub fn with_saturation(
        mut self,
        config: SaturationConfig,
    ) -> Result<Self, CompiledKokoroError> {
        config
            .validate()
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("saturation config: {e}")))?;
        let processor = SaturationProcessor::new_kokoro(config)
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("saturation: {e}")))?;
        self.saturation_config = Some(config);
        self.saturation_processor = Some(processor);
        Ok(self)
    }

    /// Whether saturation is enabled.
    #[must_use]
    pub fn has_saturation(&self) -> bool {
        self.saturation_config.is_some()
    }

    /// Access the saturation config, if set.
    #[must_use]
    pub fn saturation_config(&self) -> Option<&SaturationConfig> {
        self.saturation_config.as_ref()
    }

    /// Reset the saturation processor state.
    ///
    /// Call when starting a new synthesis session to clear decimation filter
    /// state. No-op when saturation is not enabled.
    pub fn reset_saturation(&mut self) {
        if let Some(ref mut proc) = self.saturation_processor {
            proc.reset();
        }
    }

    /// Apply multi-band dynamics processing to mixed PCM audio in place.
    ///
    /// Runs the compressor first (3-band split, per-band compression, sum),
    /// then the bus limiter as a safety ceiling. Both processors are stateful
    /// -- their envelope followers carry across calls for smooth tracking.
    ///
    /// No-op when dynamics processing is not enabled.
    pub(crate) fn apply_dynamics(&mut self, buffer: &mut [f32]) {
        if let Some(ref mut comp) = self.compressor {
            comp.process(buffer);
        }
        if let Some(ref mut lim) = self.limiter {
            lim.process(buffer);
        }
    }

    /// Mix voice PCM buffers with optional per-voice EQ, stereo imaging, and dynamics.
    ///
    /// Processing chain:
    /// 1. Per-voice EQ + de-essing (if `mix_bus` is set)
    /// 2. Stereo panning (if `stereo_config` is set) or mono mix
    /// 3. Bus EQ on the mixed output (if `mix_bus` with bus EQ is set)
    /// 4. Dynamics (compressor + limiter)
    ///
    /// Returns mono `Vec<f32>` when stereo is disabled, or interleaved stereo
    /// `[L0, R0, L1, R1, ...]` when stereo is enabled.
    pub(crate) fn mix_and_process(
        &mut self,
        voice_audio: &mut [Vec<f32>],
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        // Step 1: Per-voice EQ + de-essing.
        if let Some(ref mut bus) = self.mix_bus {
            for (i, pcm) in voice_audio.iter_mut().enumerate() {
                bus.process_voice(i, pcm);
            }
        }

        // Step 2: Mix -- stereo or mono.
        if let Some(ref stereo_cfg) = self.stereo_config {
            let (mut left, mut right) = apply_stereo_mix(voice_audio, stereo_cfg)
                .map_err(|e| CompiledKokoroError::InvalidInput(format!("stereo mix: {e}")))?;

            // Step 3: Bus EQ on each channel.
            if let Some(ref mut bus) = self.mix_bus {
                bus.process_bus(&mut left);
                bus.process_bus(&mut right);
            }

            // Step 4: Dynamics on each channel independently.
            self.apply_dynamics(&mut left);
            self.apply_dynamics(&mut right);

            // Step 4b: Saturation (after dynamics, adds harmonics).
            if let Some(ref mut sat) = self.saturation_processor {
                sat.process(&mut left);
                sat.reset(); // Reset filter state between L/R.
                sat.process(&mut right);
            }

            interleave_stereo(&left, &right)
                .map_err(|e| CompiledKokoroError::InvalidInput(format!("interleave: {e}")))
        } else {
            // Common mono path: avoid the temporary ref-slice vector and the
            // generic chorus feature-dispatch when the config is already the
            // simple gain-weighted mix case.
            let mut mixed = if can_use_simple_mono_mix(&self.config) {
                mix_voice_audio_mono_simple(voice_audio, &self.config)?
            } else {
                let voice_refs: Vec<&[f32]> =
                    voice_audio.iter().map(Vec::as_slice).collect();
                mix_voices_from_refs(&voice_refs, &self.config)?
            };

            // Step 3: Bus EQ on mono mix.
            if let Some(ref mut bus) = self.mix_bus {
                bus.process_bus(&mut mixed);
            }

            // Step 4: Dynamics.
            self.apply_dynamics(&mut mixed);

            // Step 4b: Saturation (after dynamics, adds harmonics).
            if let Some(ref mut sat) = self.saturation_processor {
                sat.process(&mut mixed);
            }

            Ok(mixed)
        }
    }

    /// Reset the dynamics compressor and limiter state.
    ///
    /// Call this when starting a new synthesis session to clear envelope
    /// follower history. No-op when dynamics is not enabled.
    pub fn reset_dynamics(&mut self) {
        if let Some(ref mut comp) = self.compressor {
            comp.reset();
        }
        if let Some(ref mut lim) = self.limiter {
            lim.reset();
        }
    }

    /// Whether dynamics processing is enabled.
    #[must_use]
    pub fn has_dynamics(&self) -> bool {
        self.dynamics_preset.is_some()
    }

    /// The active dynamics preset, if any.
    #[must_use]
    pub fn dynamics_preset(&self) -> Option<DynamicsPreset> {
        self.dynamics_preset
    }

    /// Enable the integrated chorus master pipeline.
    ///
    /// Part of #4264.
    pub fn with_chorus_pipeline(
        mut self,
        config: ChorusMasterConfig,
    ) -> Result<Self, CompiledKokoroError> {
        let pipeline = ChorusMasterPipeline::new(config)
            .map_err(|e| CompiledKokoroError::InvalidInput(format!("chorus pipeline: {e}")))?;
        self.chorus_pipeline = Some(pipeline);
        Ok(self)
    }

    /// Whether the integrated chorus master pipeline is active.
    #[must_use]
    pub fn has_chorus_pipeline(&self) -> bool {
        self.chorus_pipeline.is_some()
    }

    /// Reset the chorus master pipeline state.
    pub fn reset_chorus_pipeline(&mut self) {
        if let Some(ref mut pipeline) = self.chorus_pipeline {
            pipeline.reset();
        }
    }

    /// Mix or process voice audio through the appropriate path.
    ///
    /// Routes through either the [`ChorusMasterPipeline`] (when configured) or
    /// the default chain: detune -> humanize -> per-voice EQ/de-essing ->
    /// stereo/mono mix -> bus EQ -> dynamics (compressor + limiter).
    ///
    /// Processing state (compressor envelopes, EQ filters, limiter) persists
    /// on `self` across calls, which is essential for streaming where chunks
    /// must have smooth dynamics tracking across boundaries.
    pub(crate) fn mix_or_process(
        &mut self,
        voice_audio: &mut [Vec<f32>],
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        if let Some(ref mut pipeline) = self.chorus_pipeline {
            let (left, right) = pipeline
                .process(voice_audio)
                .map_err(|e| CompiledKokoroError::InvalidInput(format!("chorus pipeline: {e}")))?;
            let len = left.len().max(right.len());
            let mut interleaved = Vec::with_capacity(len * 2);
            for i in 0..len {
                interleaved.push(if i < left.len() { left[i] } else { 0.0 });
                interleaved.push(if i < right.len() { right[i] } else { 0.0 });
            }
            Ok(interleaved)
        } else {
            // Default path: alignment -> detune -> humanize -> mix -> dynamics.
            // Alignment FIRST: timing must be corrected before detuning or
            // humanization to avoid misaligning the corrections.
            if let Some(ref aconfig) = self.alignment_config {
                apply_alignment_in_place(voice_audio, aconfig)?;
            }
            // Pitch correction: per-voice, after alignment, before transient.
            // Snaps detected pitch toward the nearest note in the configured
            // musical scale.
            if let Some(ref pc_cfg) = self.pitch_correct_config {
                apply_pitch_correction(voice_audio, pc_cfg, KOKORO_SAMPLE_RATE as f32).map_err(
                    |e| CompiledKokoroError::InvalidInput(format!("pitch correction: {e}")),
                )?;
            }
            // Transient shaping: per-voice, after pitch correct, before detune.
            if let Some(ref transient_cfg) = self.transient_config {
                apply_transient_shaping(voice_audio, transient_cfg, KOKORO_SAMPLE_RATE as f32)
                    .map_err(|e| {
                        CompiledKokoroError::InvalidInput(format!("transient shaping: {e}"))
                    })?;
            }
            if let Some(ref dconfig) = self.detune_config {
                if let Some(ref formant_cfg) = self.formant_config {
                    // Formant-preserving path: per-voice pitch shift using
                    // the cents assigned by the detune config. Voices with
                    // >10 cents use PSOLA formant preservation; <=10 cents
                    // use fast simple pitch shift.
                    let voice_cents = dconfig.voice_cents(voice_audio.len());
                    for (i, cents) in voice_cents.iter().enumerate() {
                        if cents.abs() < 1e-6 {
                            continue; // anchor voice or zero offset
                        }
                        let rate = cents_to_rate(*cents);
                        if cents.abs() > 10.0 {
                            voice_audio[i] = shift_pitch_preserve_formant(
                                &voice_audio[i],
                                rate,
                                Some(formant_cfg),
                            )
                            .map_err(|e| {
                                CompiledKokoroError::InvalidInput(format!(
                                    "formant shift voice {i}: {e}"
                                ))
                            })?;
                        } else {
                            voice_audio[i] = simple_pitch_shift(&voice_audio[i], rate);
                        }
                    }
                } else {
                    detune_voice_pcms(voice_audio, dconfig)?;
                }
            }
            // Exciter: per-voice, after detune, adds harmonics/air.
            if let Some(ref exciter_cfg) = self.exciter_config {
                apply_exciter(voice_audio, exciter_cfg, KOKORO_SAMPLE_RATE as f32)
                    .map_err(|e| CompiledKokoroError::InvalidInput(format!("exciter: {e}")))?;
            }
            if let Some(ref hconfig) = self.humanize_config {
                humanize_voice_pcms(voice_audio, hconfig)?;
            }
            // Doubler / ADT: per-voice, after humanize, creates doubled
            // copies with timing/pitch variation.
            if let Some(ref doubler_cfg) = self.doubler_config {
                apply_doubler_per_voice(voice_audio, doubler_cfg, KOKORO_SAMPLE_RATE as f32)
                    .map_err(|e| CompiledKokoroError::InvalidInput(format!("doubler: {e}")))?;
            }
            // Breath insertion: per-voice, after humanize + doubler, before mix.
            if let (Some(ref bconfig), Some(ref mut bgen)) =
                (&self.breath_config, &mut self.breath_generator)
            {
                if let Some(reference_voice) = voice_audio.first() {
                    let pauses = detect_pauses(reference_voice, bconfig);
                    if !pauses.is_empty() {
                        insert_breath_sounds(voice_audio, &pauses, bgen, bconfig).map_err(|e| {
                            CompiledKokoroError::InvalidInput(format!("breath insertion: {e}"))
                        })?;
                    }
                }
            }
            // Voice bleed: per-voice crosstalk, after breath, before spatial/mix.
            if let Some(ref bleed_cfg) = self.bleed_config {
                apply_voice_bleed(voice_audio, bleed_cfg, KOKORO_SAMPLE_RATE)
                    .map_err(|e| CompiledKokoroError::InvalidInput(format!("voice bleed: {e}")))?;
            }
            // Ducking: per-voice, after bleed, lead voice prominence.
            if let (Some(ref ducking_cfg), Some(ref mut ducker)) =
                (&self.ducking_config, &mut self.ducker)
            {
                ducker
                    .process(voice_audio, ducking_cfg)
                    .map_err(|e| CompiledKokoroError::InvalidInput(format!("ducking: {e}")))?;
            }
            // When spatial is enabled, process each voice through distance
            // attenuation + ILD stereo panning, sum into L/R, apply dynamics,
            // and return interleaved stereo. This bypasses mix_and_process.
            if let Some(ref spatial_cfg) = self.spatial_config {
                let n = voice_audio.len();
                let positions = auto_layout_spatial(n, spatial_cfg).map_err(|e| {
                    CompiledKokoroError::InvalidInput(format!("spatial layout: {e}"))
                })?;
                let max_len = voice_audio.iter().map(Vec::len).max().unwrap_or(0);
                let mut left = vec![0.0f32; max_len];
                let mut right = vec![0.0f32; max_len];
                for (voice, pos) in voice_audio.iter().zip(positions.iter()) {
                    let (vl, vr) = process_voice_spatial(voice, spatial_cfg, pos).map_err(|e| {
                        CompiledKokoroError::InvalidInput(format!("spatial processing: {e}"))
                    })?;
                    for (i, (&sl, &sr)) in vl.iter().zip(vr.iter()).enumerate() {
                        left[i] += sl;
                        right[i] += sr;
                    }
                }
                // Apply dynamics to each channel.
                self.apply_dynamics(&mut left);
                self.apply_dynamics(&mut right);
                interleave_stereo(&left, &right)
                    .map_err(|e| CompiledKokoroError::InvalidInput(format!("interleave: {e}")))
            } else {
                self.mix_and_process(voice_audio)
            }
        }
    }

    /// Synthesize all voices and mix into a single PCM buffer.
    ///
    /// Each voice synthesizes its own `(input_ids, style)` pair independently,
    /// then all outputs are mixed using the chorus config's per-voice gains.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Per-voice token IDs. Length must equal `n_voices`.
    ///   Each element is shape `[1, T]`.
    /// * `styles` - Per-voice style embeddings. Length must equal `n_voices`.
    ///   Each element is shape `[1, 2*style_dim]`.
    /// * `speed` - Speaking rate multiplier (shared across all voices).
    /// * `cache` - Metal pipeline cache (shared across all voices).
    ///
    /// # Returns
    ///
    /// Mixed PCM audio at 24kHz mono. Length equals the longest voice's output.
    ///
    /// # Errors
    ///
    /// Returns error if any voice fails to synthesize, or if input/style
    /// counts don't match the voice count.
    pub fn synthesize_chorus(
        &mut self,
        inputs: &[DynTensor],
        styles: &[DynTensor],
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        let n = self.voices.len();
        if inputs.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "inputs length {} != n_voices {n}",
                inputs.len()
            ))
            .into());
        }
        if styles.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))
            .into());
        }
        validate_speed(speed)?;
        validate_indexed_input_ids(
            inputs,
            self.voices[0].config().plbert.max_position_embeddings,
            "inputs",
        )?;

        // Pre-size the arena for the N-voice synthesis loop (#4289).
        // Multiply by 2 to account for Phase 1 regulate accumulation across
        // all voices plus one decode pass. Per-voice checkpoint/restore in
        // Phase 2 prevents further growth. Part of dvoice#2420.
        let arena_estimate = self.voices[0].estimate_arena_bytes().saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        // Synthesize each voice sequentially.
        // GPU work is pipelined by Metal — each synthesize() call encodes
        // into the lazy command buffer, and GPU overlap happens naturally.
        // Per-voice arena checkpoint/restore to prevent overflow thrashing
        // across N voices. Part of dvoice#2420.
        let mut voice_audio: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (i, voice) in self.voices.iter_mut().enumerate() {
            let _arena_cp = DefaultArenaCheckpoint::new();
            let (audio_tensor, _cert) = voice.synthesize(&inputs[i], &styles[i], speed, cache)?;
            // Extract PCM from [1, 1, T] tensor → flat Vec<f32>.
            let pcm = extract_pcm_from_audio(&audio_tensor)?;
            voice_audio.push(pcm);
        }

        // Unified mixing path: detune -> humanize -> EQ -> mix -> dynamics.
        self.mix_or_process(&mut voice_audio)
    }

    /// Synthesize all voices with the same text but different styles.
    ///
    /// Convenience method for the common chorus pattern: same lyrics,
    /// different speaker voices. Delegates to [`synthesize_chorus_shared_encode`]
    /// which runs encoding once and reuses the result for all voices.
    pub fn synthesize_chorus_same_text(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        self.synthesize_chorus_shared_encode(input_ids, styles, speed, cache)
    }

    /// Synthesize all voices with shared text encoding (same text, different styles).
    ///
    /// Steps 1-2 (PLBert + TextEncoder) execute once on `voices[0]`. Steps 3-8
    /// execute per-voice using the shared encoding results. For 8 voices with
    /// the same text, this eliminates 7 redundant encoding passes (~66ms GPU
    /// at D=512 SEQ_LEN=32, per #3229 profiling).
    ///
    /// GPU data flow: `step_encode` produces standalone GPU buffers (via
    /// `without_arena`) that survive across all per-voice iterations. Each
    /// voice's Steps 3-8 encode GPU work into the lazy batch; the only hot-path
    /// sync points are `step_regulate`'s 4-byte scalar readback and the
    /// per-voice pipeline-exit `to_device(&cpu())` transfer.
    ///
    /// Part of #3351 (T4.2), design: `designs/2026-03-23-multi-voice-gpu-scheduling.md`.
    pub fn synthesize_chorus_shared_encode(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        let n = self.voices.len();
        if styles.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))
            .into());
        }
        validate_speed(speed)?;
        validate_input_ids(
            input_ids,
            self.voices[0].config().plbert.max_position_embeddings,
        )?;

        // Reclaim buffer pool entries from previous synthesis call (#3079 D3).
        crate::arena::pool_reclaim();

        // Pre-size the arena for the N-voice decode loop (#4289).
        // Multiply by 2 to account for Phase 1 regulate accumulation across
        // all voices plus one decode pass. Part of dvoice#2420.
        let arena_estimate = self.voices[0].estimate_arena_bytes().saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        // Pre-split styles and upload to GPU (outside NaN skip scope).
        // All voices share the same style_dim (cloned from primary), so
        // voices[0].split_style works for all styles.
        let mut decoder_styles: Vec<DynTensor> = Vec::with_capacity(n);
        let mut prosody_styles: Vec<DynTensor> = Vec::with_capacity(n);
        for style in styles {
            let split = self.voices[0].split_style(style)?;
            decoder_styles.push(split.decoder_style.to_device(&gpu())?);
            prosody_styles.push(split.prosody_style.to_device(&gpu())?);
        }

        let result = (|| -> Result<Vec<f32>, CompiledKokoroError> {
            // Steps 1-8 inside NaN Skip scope (matches synthesize() convention).
            let audio_tensors: Vec<DynTensor> = with_nan_check_policy(
                NanCheckPolicy::Skip,
                || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                    // Steps 1-2: Encode once on voices[0].
                    let enc = self.voices[0].step_encode(input_ids, cache)?;

                    // Two-phase per-voice pipeline (#4290):
                    // Phase 1: Steps 3-4 (prosody + regulate) for all voices.
                    //   step_regulate has a GPU submit+sync for prefix-sum readback.
                    //   Batching this phase runs all N syncs without interleaving
                    //   with the heavy decode GPU work.
                    // Phase 2: Steps 5-8 (f0 → harmonic → generate → iSTFT) for
                    //   all voices. Purely GPU — no syncs. Metal's lazy command
                    //   batching pipelines all voices' decode work into one batch.
                    let mut regulate_results: Vec<super::StepRegulateResult> =
                        Vec::with_capacity(n);
                    for (i, voice) in self.voices.iter_mut().enumerate() {
                        let pros = voice.step_predict_prosody(
                            &enc.bert_features,
                            &prosody_styles[i],
                            enc.seq_len,
                            cache,
                        )?;
                        let reg = voice.step_regulate(
                            &pros.dur_logits,
                            &pros.features,
                            &enc.text_features,
                            speed,
                            cache,
                        )?;
                        regulate_results.push(reg);
                    }

                    // Phase 2: Steps 5-8 (sync-free decode) for all voices.
                    // Uses run_decode_phase which inserts GpuFence submits
                    // between f0, harmonic, generator, and iSTFT segments,
                    // enabling CPU-GPU overlap within each voice's decode.
                    // Per-voice arena checkpoint/restore: each voice's decode
                    // intermediates are reclaimed after the output is consumed
                    // (blitted to standalone via verify_and_extract_pcm).
                    // Without this, N voices accumulate arena memory causing
                    // overflow thrashing (64MB -> 128MB -> 256MB -> 512MB+).
                    // Part of dvoice#2420, #4264.
                    let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                    for (i, voice) in self.voices.iter_mut().enumerate() {
                        let _arena_cp = DefaultArenaCheckpoint::new();
                        let audio = run_decode_phase(
                            voice,
                            &regulate_results[i],
                            &prosody_styles[i],
                            &decoder_styles[i],
                            cache,
                        )?;
                        audios.push(audio);
                    }
                    Ok(audios)
                },
            )?;

            // Outside Skip scope: NaN guard + verify per voice.
            let mut voice_audio = verify_and_extract_pcm(&self.voices, &audio_tensors)?;

            // Unified mixing path: detune -> humanize -> EQ -> mix -> dynamics.
            self.mix_or_process(&mut voice_audio)
        })();

        // Clean up stale GPU commands on error (matches synthesize() pattern).
        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }

    /// Synthesize all voices with the same text and same style but varied speeds.
    ///
    /// Uses shared encoding: Steps 1-2 execute once, Steps 3-8 execute per-voice
    /// with the per-voice speed. Encoding is speed-independent, so this
    /// eliminates (N-1) redundant encoding passes.
    ///
    /// Useful for creating a "thickened" chorus where slight speed variations
    /// produce a natural ensemble detuning effect.
    pub fn synthesize_chorus_varied_speed(
        &mut self,
        input_ids: &DynTensor,
        style: &DynTensor,
        speeds: &[f32],
        cache: &PipelineCache,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        let n = self.voices.len();
        if speeds.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "speeds length {} != n_voices {n}",
                speeds.len()
            ))
            .into());
        }
        for &s in speeds {
            validate_speed(s)?;
        }
        validate_input_ids(
            input_ids,
            self.voices[0].config().plbert.max_position_embeddings,
        )?;

        // Reclaim buffer pool entries from previous synthesis call.
        crate::arena::pool_reclaim();

        // Pre-size the arena for the N-voice decode loop (#4289).
        // Multiply by 2 to account for Phase 1 regulate accumulation across
        // all voices plus one decode pass. Part of dvoice#2420.
        let arena_estimate = self.voices[0].estimate_arena_bytes().saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        // Pre-split style once (all voices use the same style).
        let split = self.voices[0].split_style(style)?;
        let decoder_style = split.decoder_style.to_device(&gpu())?;
        let prosody_style = split.prosody_style.to_device(&gpu())?;

        let result = (|| -> Result<Vec<f32>, CompiledKokoroError> {
            let audio_tensors: Vec<DynTensor> = with_nan_check_policy(
                NanCheckPolicy::Skip,
                || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                    // Steps 1-2: Encode once on voices[0].
                    let enc = self.voices[0].step_encode(input_ids, cache)?;

                    // Two-phase per-voice pipeline (#4290):
                    // Phase 1: Steps 3-4 (prosody + regulate, has GPU sync).
                    let mut regulate_results: Vec<super::StepRegulateResult> =
                        Vec::with_capacity(n);
                    for (i, voice) in self.voices.iter_mut().enumerate() {
                        let pros = voice.step_predict_prosody(
                            &enc.bert_features,
                            &prosody_style,
                            enc.seq_len,
                            cache,
                        )?;
                        let reg = voice.step_regulate(
                            &pros.dur_logits,
                            &pros.features,
                            &enc.text_features,
                            speeds[i],
                            cache,
                        )?;
                        regulate_results.push(reg);
                    }

                    // Phase 2: Steps 5-8 (sync-free decode).
                    // Uses run_decode_phase which inserts GpuFence submits
                    // between f0, harmonic, generator, and iSTFT segments.
                    // Per-voice arena checkpoint/restore to prevent overflow
                    // thrashing across N voices. Part of dvoice#2420, #4264.
                    let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                    for (i, voice) in self.voices.iter_mut().enumerate() {
                        let _arena_cp = DefaultArenaCheckpoint::new();
                        let audio = run_decode_phase(
                            voice,
                            &regulate_results[i],
                            &prosody_style,
                            &decoder_style,
                            cache,
                        )?;
                        audios.push(audio);
                    }
                    Ok(audios)
                },
            )?;

            // Outside Skip scope: NaN guard + verify per voice.
            let mut voice_audio = verify_and_extract_pcm(&self.voices, &audio_tensors)?;

            // Unified mixing path: detune -> humanize -> EQ -> mix -> dynamics.
            self.mix_or_process(&mut voice_audio)
        })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }

    /// Synthesize all voices with shared encoding and GpuFence-pipelined decode.
    ///
    /// Functionally identical to [`synthesize_chorus_shared_encode`] -- produces
    /// identical mixed audio. The difference is GPU scheduling: after each
    /// voice's Steps 3-8 encode GPU commands, the pending work is submitted
    /// non-blocking via [`GpuFence::submit_current`]. This lets the GPU execute
    /// voice N's post-regulate work (steps 5-8) while the CPU encodes voice
    /// N+1's command buffers.
    ///
    /// # Pipelining strategy
    ///
    /// ```text
    /// GPU:  |--voice0 steps5-8--|--voice1 steps5-8--|--voice2 steps5-8--|
    /// CPU:     |voice1 step3-4|    |voice2 step3-4|    |voice3 step3-4|
    /// ```
    ///
    /// Within each voice, `step_regulate` (step 4) has an inherent
    /// `submit()+sync()` for the 4-byte prefix-sum scalar readback. This is
    /// unavoidable. The pipelining win is overlapping the *post-regulate*
    /// GPU execution of voice N with the *pre-regulate* CPU encoding of
    /// voice N+1.
    ///
    /// For 4 voices, this can reduce wall-clock time by ~25-40% depending
    /// on GPU utilization and segment complexity.
    ///
    /// # Arena safety
    ///
    /// [`GpuFence::wait`] does NOT reset the activation arena. This is safe
    /// because all step outputs use `to_standalone()` or `without_arena`
    /// wrappers, producing standalone GPU buffers immune to arena resets.
    /// The final `verify_and_extract_pcm` calls `to_device(&cpu())` which
    /// triggers a flush, resetting the arena after all work completes.
    ///
    /// Part of #4290.
    pub fn synthesize_chorus_pipelined(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        let n = self.voices.len();
        if styles.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))
            .into());
        }
        validate_speed(speed)?;
        validate_input_ids(
            input_ids,
            self.voices[0].config().plbert.max_position_embeddings,
        )?;

        crate::arena::pool_reclaim();

        // Pre-size the arena for the N-voice decode loop (#4289).
        // Multiply by 2 to account for Phase 1 regulate accumulation across
        // all voices plus one decode pass. Part of dvoice#2420.
        let arena_estimate = self.voices[0].estimate_arena_bytes().saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        let mut decoder_styles: Vec<DynTensor> = Vec::with_capacity(n);
        let mut prosody_styles: Vec<DynTensor> = Vec::with_capacity(n);
        for style in styles {
            let split = self.voices[0].split_style(style)?;
            decoder_styles.push(split.decoder_style.to_device(&gpu())?);
            prosody_styles.push(split.prosody_style.to_device(&gpu())?);
        }

        let result = (|| -> Result<Vec<f32>, CompiledKokoroError> {
            let audio_tensors: Vec<DynTensor> = with_nan_check_policy(
                NanCheckPolicy::Skip,
                || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                    let enc = self.voices[0].step_encode(input_ids, cache)?;

                    // Submit encode GPU work, wait for completion.
                    let encode_fence = GpuFence::submit_current()
                        .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                    if let Some(f) = encode_fence {
                        f.wait()
                            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                    }

                    let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                    let mut prev_fence: Option<GpuFence> = None;

                    for (i, voice) in self.voices.iter_mut().enumerate() {
                        // Per-voice arena checkpoint/restore to prevent overflow
                        // thrashing across N voices. Part of dvoice#2420.
                        let _arena_cp = DefaultArenaCheckpoint::new();
                        let audio = run_voice_pipeline(
                            voice,
                            &enc,
                            &prosody_styles[i],
                            &decoder_styles[i],
                            speed,
                            cache,
                        )?;
                        audios.push(audio);

                        // Submit this voice's post-regulate GPU work non-blocking.
                        let fence = GpuFence::submit_current()
                            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;

                        // Wait for the *previous* voice's fence after submitting
                        // the current voice, maximizing GPU overlap.
                        if let Some(f) = prev_fence.take() {
                            f.wait()
                                .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                        }
                        prev_fence = fence;
                    }

                    if let Some(f) = prev_fence {
                        f.wait()
                            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                    }

                    Ok(audios)
                },
            )?;

            let mut voice_audio = verify_and_extract_pcm(&self.voices, &audio_tensors)?;

            // Unified mixing path: detune -> humanize -> EQ -> mix -> dynamics.
            self.mix_or_process(&mut voice_audio)
        })();

        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }

    /// Synthesize all voices with shared encoding and bulk-parallel GPU decode.
    ///
    /// This is the highest-throughput chorus method. Like
    /// [`synthesize_chorus_pipelined`](Self::synthesize_chorus_pipelined), encoding
    /// runs once (steps 1-2) and the regulate phase (steps 3-4) runs sequentially
    /// per-voice (because `step_regulate` has an unavoidable GPU sync for
    /// prefix-sum readback). The key difference is in the decode phase:
    ///
    /// - **Pipelined** (existing): submit voice N, wait for voice N-1, encode voice N+1.
    /// - **Parallel** (this method): encode ALL voices' decode, submit each as a
    ///   separate command buffer, then wait for ALL fences at the end.
    ///
    /// # GPU scheduling
    ///
    /// ```text
    /// Phase 1 (sequential -- has GPU syncs):
    ///   voice0: steps 3-4 (prosody + regulate + sync)
    ///   voice1: steps 3-4 (prosody + regulate + sync)
    ///   ...
    ///
    /// Phase 2 (parallel GPU -- no syncs until end):
    ///   voice0: steps 5-8 -> submit_current() -> fence0
    ///   voice1: steps 5-8 -> submit_current() -> fence1
    ///   ...
    ///   wait(fence0), wait(fence1), ...
    /// ```
    ///
    /// Metal can execute the N command buffers concurrently across its GPU
    /// cores (M4 Max has 40 cores). For 4-8 voices, this can improve GPU
    /// utilization significantly compared to the sequential pipelining
    /// approach.
    ///
    /// # Arena safety
    ///
    /// Each voice's decode outputs use `to_standalone()` or `without_arena`
    /// wrappers, producing standalone GPU buffers that survive arena resets.
    /// The fences do NOT reset the arena (see [`GpuFence`] docs). The final
    /// `verify_and_extract_pcm` call triggers `to_device(&cpu())` which
    /// flushes and resets the arena after all work completes.
    ///
    /// # CompiledKokoro is !Send
    ///
    /// All work runs on the calling thread. "Parallel" refers to GPU command
    /// buffer interleaving, not CPU thread parallelism. Metal's command queue
    /// executes submitted command buffers with hardware-level overlap.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use nn_metal::compiled_kokoro::chorus::KokoroChorus;
    /// use nn_models::kokoro_chorus::ChorusConfig;
    ///
    /// let config = ChorusConfig::equal_gain(4)?;
    /// let mut chorus = KokoroChorus::new(&primary_kokoro, config)?;
    /// let audio = chorus.synthesize_chorus_parallel(&input_ids, &styles, 1.0, &cache)?;
    /// // `audio` is mixed PCM at 24kHz — ready for playback.
    /// ```
    ///
    /// Part of #4290.
    pub fn synthesize_chorus_parallel(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<Vec<f32>, CompiledKokoroError> {
        let n = self.voices.len();
        if styles.len() != n {
            return Err(KokoroError::InvalidInput(format!(
                "styles length {} != n_voices {n}",
                styles.len()
            ))
            .into());
        }
        validate_speed(speed)?;
        validate_input_ids(
            input_ids,
            self.voices[0].config().plbert.max_position_embeddings,
        )?;

        crate::arena::pool_reclaim();

        // Pre-size the arena for the N-voice decode loop (#4289).
        // Multiply by 2 to account for Phase 1 regulate accumulation across
        // all voices plus one decode pass. Part of dvoice#2420.
        let arena_estimate = self.voices[0].estimate_arena_bytes().saturating_mul(2);
        if arena_estimate > 0 {
            let _ = crate::arena::ensure_default_arena_capacity(cache.context(), arena_estimate);
        }

        // Pre-split styles and upload to GPU (outside NaN skip scope).
        let mut decoder_styles: Vec<DynTensor> = Vec::with_capacity(n);
        let mut prosody_styles: Vec<DynTensor> = Vec::with_capacity(n);
        for style in styles {
            let split = self.voices[0].split_style(style)?;
            decoder_styles.push(split.decoder_style.to_device(&gpu())?);
            prosody_styles.push(split.prosody_style.to_device(&gpu())?);
        }

        let result = (|| -> Result<Vec<f32>, CompiledKokoroError> {
            let audio_tensors: Vec<DynTensor> = with_nan_check_policy(
                NanCheckPolicy::Skip,
                || -> Result<Vec<DynTensor>, CompiledKokoroError> {
                    // Steps 1-2: Encode once on voices[0].
                    let enc = self.voices[0].step_encode(input_ids, cache)?;

                    // Submit encode GPU work and wait -- subsequent phases depend
                    // on encode results being GPU-resident.
                    let encode_fence = GpuFence::submit_current()
                        .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                    if let Some(f) = encode_fence {
                        f.wait()
                            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                    }

                    // Phase 1: Steps 3-4 (prosody + regulate) sequentially.
                    // step_regulate has a GPU submit+sync for prefix-sum readback;
                    // this is unavoidable and must complete before Phase 2.
                    let mut regulate_results: Vec<super::StepRegulateResult> =
                        Vec::with_capacity(n);
                    for (i, voice) in self.voices.iter_mut().enumerate() {
                        let pros = voice.step_predict_prosody(
                            &enc.bert_features,
                            &prosody_styles[i],
                            enc.seq_len,
                            cache,
                        )?;
                        let reg = voice.step_regulate(
                            &pros.dur_logits,
                            &pros.features,
                            &enc.text_features,
                            speed,
                            cache,
                        )?;
                        regulate_results.push(reg);
                    }

                    // Phase 2: Steps 5-8 (decode) for all voices. Each voice's
                    // GPU work is submitted as a separate command buffer via
                    // GpuFence::submit_current(), enabling Metal to execute
                    // multiple command buffers concurrently.
                    // Per-voice arena checkpoint/restore to prevent overflow
                    // thrashing across N voices. Part of dvoice#2420.
                    let mut audios: Vec<DynTensor> = Vec::with_capacity(n);
                    let mut fences: Vec<Option<GpuFence>> = Vec::with_capacity(n);
                    for (i, voice) in self.voices.iter_mut().enumerate() {
                        let _arena_cp = DefaultArenaCheckpoint::new();
                        let (audio, fence) = run_voice_decode_async(
                            voice,
                            &regulate_results[i],
                            &prosody_styles[i],
                            &decoder_styles[i],
                            cache,
                        )?;
                        audios.push(audio);
                        fences.push(fence);
                    }

                    // Wait for ALL fences. The GPU may have executed them
                    // concurrently across its compute cores.
                    for f in fences.into_iter().flatten() {
                        f.wait()
                            .map_err(|e| CompiledKokoroError::Tensor(Box::new(e)))?;
                    }

                    Ok(audios)
                },
            )?;

            // Outside Skip scope: NaN guard + verify per voice.
            let mut voice_audio = verify_and_extract_pcm(&self.voices, &audio_tensors)?;

            // Unified mixing path: detune -> humanize -> EQ -> mix -> dynamics.
            self.mix_or_process(&mut voice_audio)
        })();

        // Clean up stale GPU commands on error (matches synthesize() pattern).
        if result.is_err() {
            crate::gpu_scope::discard_pending_batch();
        }

        result
    }

    // -- Arena report ---------------------------------------------------------

    /// Synthesize chorus with per-synthesis arena utilization tracking.
    ///
    /// Wraps [`synthesize_chorus_shared_encode`] with arena stats capture.
    /// The returned [`KokoroArenaReport`] shows how many GPU buffers were
    /// reused via the arena vs. freshly allocated across all N voices.
    ///
    /// # Arguments
    ///
    /// Same as [`synthesize_chorus_shared_encode`].
    ///
    /// # Returns
    ///
    /// `(mixed_audio, arena_report)` -- mixed PCM at 24kHz plus per-synthesis
    /// allocation metrics.
    ///
    /// Part of #4264.
    pub fn synthesize_chorus_with_arena_report(
        &mut self,
        input_ids: &DynTensor,
        styles: &[DynTensor],
        speed: f32,
        cache: &PipelineCache,
    ) -> Result<(Vec<f32>, super::KokoroArenaReport), CompiledKokoroError> {
        let pre = super::arena_report::snapshot_arena_pre();
        let audio = self.synthesize_chorus_shared_encode(input_ids, styles, speed, cache)?;
        let report = super::arena_report::build_arena_report(&pre);
        Ok((audio, report))
    }

    // -- Diagnostics --------------------------------------------------------

    /// Get the `Arc` refcount of the shared state.
    ///
    /// Returns the strong count of all `Arc` references to the shared weights,
    /// including the chorus voices and any external references (e.g., the
    /// primary instance the chorus was cloned from). Useful for memory diagnostics.
    #[must_use]
    pub fn shared_state_refcount(&self) -> usize {
        self.voices
            .first()
            .map_or(0, CompiledKokoro::shared_state_refcount)
    }

    /// Total GPU weight bytes across all voices (should be same as a single voice
    /// since weights are aliased, not copied).
    #[must_use]
    pub fn gpu_weight_bytes_per_voice(&self) -> usize {
        self.voices
            .first()
            .map_or(0, CompiledKokoro::gpu_weight_bytes)
    }
}

#[cfg(test)]
#[path = "compiled_kokoro_chorus_tests.rs"]
mod chorus_tests;

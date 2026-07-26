// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended preset library with validation and comparison for Kokoro chorus.
//!
//! Provides named production-ready presets beyond the basic presets in
//! [`ChorusMasterConfig`], each tuned for a specific musical/audio context.
//! Also provides preset validation (safe range checking) and a comparison
//! tool for diffing two preset configurations.
//!
//! # Presets
//!
//! | Preset      | Voices | Stereo | Vibrato | Reverb | Dynamics     | Use case                     |
//! |-------------|--------|--------|---------|--------|--------------|------------------------------|
//! | `ACapella`  | any    | 0.70   | light   | none   | gentle       | Tight, dry, unaccompanied    |
//! | `Cathedral` | any    | 1.00   | deep    | large  | broadcast    | Large hall, wide, breathy    |
//! | `Studio`    | any    | 0.60   | subtle  | short  | mastering    | Professional multitrack      |
//! | `LiveStage` | any    | 0.75   | natural | medium | broadcast    | Moderate reverb, natural     |
//! | `Intimate`  | any    | 0.25   | none    | none   | mastering    | Close-mic, minimal           |
//! | `Epic`      | 16+    | 1.00   | deep    | large  | aggressive   | Maximum voices, heavy        |
//! | `Whisper`   | any    | 0.20   | none    | tiny   | gentle       | Very soft, ASMR-like         |
//! | `Choir`     | any    | 0.80   | natural | medium | gentle       | Traditional choral blend     |
//!
//! # Usage
//!
//! ```rust,no_run
//! use nn_models::kokoro_chorus_preset_library::{ChorusPreset, validate_preset};
//!
//! let config = ChorusPreset::Cathedral.to_config(8).unwrap();
//! validate_preset(&config).expect("cathedral preset is valid");
//! ```
//!
//! Part of #4264, #3351.

use crate::kokoro_chorus_blend::EnsembleBlendConfig;
use crate::kokoro_chorus_detune::{DetuneConfig, DetuneDistribution};
use crate::kokoro_chorus_dynamics::DynamicsPreset;
use crate::kokoro_chorus_eq::{DeEsserConfig, EqConfig};
use crate::kokoro_chorus_humanize::HumanizeConfig;
use crate::kokoro_chorus_pipeline::ChorusMasterConfig;
use crate::kokoro_chorus_reverb::ReverbConfig;
use crate::kokoro_chorus_stereo::StereoChorusConfig;
use crate::kokoro_chorus_vibrato::VibratoConfig;
use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// ChorusPreset enum
// ---------------------------------------------------------------------------

/// Named production-ready chorus presets.
///
/// Each variant provides a fully-configured [`ChorusMasterConfig`] tuned for
/// a specific musical or audio context. Call [`to_config`](Self::to_config)
/// with the desired voice count to get the config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChorusPreset {
    /// Tight, dry, focused chorus for unaccompanied singing.
    ///
    /// No reverb, moderate stereo width, light vibrato with narrow spread,
    /// gentle dynamics to preserve natural vocal dynamics. Optimized for
    /// a cappella groups where every voice must be distinct and present.
    ACapella,

    /// Large reverb, wide stereo, breathy cathedral sound.
    ///
    /// Maximum stereo width, deep vibrato with wide spread, large-hall
    /// reverb with long tails, high-pass at 120 Hz to keep low end clean.
    /// Broadcast dynamics to control reverb mud. Sounds like a choir in
    /// a stone cathedral.
    Cathedral,

    /// Controlled, warm, professional multitrack studio sound.
    ///
    /// Moderate stereo, subtle vibrato, short room reverb, mastering-grade
    /// transparent dynamics. Warm EQ with gentle presence lift. The sound
    /// of a well-produced studio vocal session.
    Studio,

    /// Moderate reverb, natural spacing for a live performance feel.
    ///
    /// Natural vibrato rate, medium room reverb, balanced stereo spread.
    /// Broadcast dynamics for even loudness. Simulates a live concert
    /// stage with natural acoustic space.
    LiveStage,

    /// Close-mic'd, minimal processing, ASMR-like proximity.
    ///
    /// Very narrow stereo, no vibrato, no reverb, warm EQ with proximity
    /// bass boost, mastering-grade transparent dynamics. Creates the feel
    /// of whispered voices very close to the listener.
    Intimate,

    /// Maximum voices, heavy dynamics, wide spatial field.
    ///
    /// Full stereo width, deep vibrato with maximum spread, large reverb,
    /// aggressive dynamics compression, wide Gaussian detuning. Designed
    /// for 16+ voices creating a massive, cinematic wall of sound.
    Epic,

    /// Very soft, close, ASMR-like whisper chorus.
    ///
    /// Extremely narrow stereo, no vibrato, tiny reverb for subtle air,
    /// gentle dynamics, very warm EQ rolling off highs aggressively.
    /// Soft, breathy, intimate. Minimal detuning to avoid phasing.
    Whisper,

    /// Traditional choral sound with balanced blend.
    ///
    /// Wide stereo, natural singing vibrato, medium reverb, gentle dynamics.
    /// Gaussian detuning for natural intonation spread. Strong ensemble
    /// blending with formant preservation. The classic choir sound.
    Choir,
}

impl ChorusPreset {
    /// All available presets, in definition order.
    pub const ALL: &'static [Self] = &[
        Self::ACapella,
        Self::Cathedral,
        Self::Studio,
        Self::LiveStage,
        Self::Intimate,
        Self::Epic,
        Self::Whisper,
        Self::Choir,
    ];

    /// Human-readable name for this preset.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::ACapella => "A Cappella",
            Self::Cathedral => "Cathedral",
            Self::Studio => "Studio",
            Self::LiveStage => "Live Stage",
            Self::Intimate => "Intimate",
            Self::Epic => "Epic",
            Self::Whisper => "Whisper",
            Self::Choir => "Choir",
        }
    }

    /// Short description of the preset's character.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::ACapella => "Tight, dry, focused for unaccompanied singing",
            Self::Cathedral => "Large reverb, wide stereo, breathy hall sound",
            Self::Studio => "Controlled, warm, professional multitrack",
            Self::LiveStage => "Moderate reverb, natural live performance spacing",
            Self::Intimate => "Close-mic, minimal processing, proximity warmth",
            Self::Epic => "Maximum voices, heavy dynamics, wide spatial field",
            Self::Whisper => "Very soft, close, ASMR-like whisper",
            Self::Choir => "Traditional choral sound, balanced blend",
        }
    }

    /// Convert this preset to a [`ChorusMasterConfig`] for `n_voices`.
    ///
    /// # Errors
    ///
    /// Returns [`KokoroError::InvalidConfig`] if `n_voices` is 0 or > 32.
    pub fn to_config(self, n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
        match self {
            Self::ACapella => build_acappella(n_voices),
            Self::Cathedral => build_cathedral(n_voices),
            Self::Studio => build_studio(n_voices),
            Self::LiveStage => build_live_stage(n_voices),
            Self::Intimate => build_intimate(n_voices),
            Self::Epic => build_epic(n_voices),
            Self::Whisper => build_whisper(n_voices),
            Self::Choir => build_choir(n_voices),
        }
    }
}

// ---------------------------------------------------------------------------
// Preset builders
// ---------------------------------------------------------------------------

fn build_acappella(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Presence-forward EQ: slight low cut to reduce muddiness when voices
    // stack, mid presence boost for clarity, gentle high shelf.
    config.eq = Some(EqConfig {
        low_freq: 150.0,
        low_gain_db: -1.0,
        mid_freq: 3000.0,
        mid_gain_db: 1.5,
        mid_q: 1.0,
        high_freq: 9000.0,
        high_gain_db: -0.5,
    });

    // Moderate de-essing: stacked voices amplify sibilance.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.2,
        threshold_db: -20.0,
        max_reduction_db: -10.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    });

    // Light vibrato: subtle shimmer without obvious pitch wobble.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.5,
        depth_cents: 15.0,
        rate_spread_hz: 0.3,
        depth_spread_cents: 5.0,
        onset_sec: 0.15,
    });

    // Moderate detuning for ensemble width, uniform distribution.
    config.detune = Some(DetuneConfig {
        cents_spread: 8.0,
        distribution: DetuneDistribution::Uniform,
        seed: 0,
    });

    // Full humanization for natural a cappella feel.
    config.humanize = Some(HumanizeConfig::default());

    // Moderate ensemble blending.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.4,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Moderate stereo: wide enough for separation, not so wide it feels diffuse.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.70));

    // Gentle dynamics: preserve vocal expression.
    config.dynamics = Some(DynamicsPreset::Gentle.to_config());

    // No reverb: dry, focused a cappella sound.
    config.reverb = None;
    config.limiter_enabled = true;

    Ok(config)
}

fn build_cathedral(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // High-pass at 120 Hz to keep low end clean in the reverb tail,
    // slight presence dip to avoid harshness in the decay.
    config.eq = Some(EqConfig {
        low_freq: 120.0,
        low_gain_db: -3.0,
        mid_freq: 2500.0,
        mid_gain_db: -1.5,
        mid_q: 0.7,
        high_freq: 8000.0,
        high_gain_db: -0.5,
    });

    // De-essing: reverb amplifies sibilance.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.0,
        threshold_db: -18.0,
        max_reduction_db: -10.0,
        attack_sec: 0.001,
        release_sec: 0.060,
    });

    // Deep vibrato: dramatic choir shimmer with wide per-voice spread.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.0,
        depth_cents: 45.0,
        rate_spread_hz: 0.6,
        depth_spread_cents: 15.0,
        onset_sec: 0.25,
    });

    // Wide Gaussian detuning for thick ensemble texture.
    config.detune = Some(DetuneConfig {
        cents_spread: 16.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Full humanization.
    config.humanize = Some(HumanizeConfig::default());

    // Strong blending for cohesive choir unit.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.6,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Maximum stereo width for immersive spatial image.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(1.0));

    // Broadcast dynamics to keep reverb tails controlled.
    config.dynamics = Some(DynamicsPreset::Broadcast.to_config());

    // Large hall reverb: high wet mix, large room, warm damping.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.35,
        room_size: 0.85,
        early_reflections: true,
        damping: 0.6,
    });
    config.limiter_enabled = true;

    Ok(config)
}

fn build_studio(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Warm, controlled EQ: gentle low warmth, subtle presence at 2.5kHz,
    // smooth highs.
    config.eq = Some(EqConfig {
        low_freq: 200.0,
        low_gain_db: 1.0,
        mid_freq: 2500.0,
        mid_gain_db: 1.5,
        mid_q: 0.9,
        high_freq: 8500.0,
        high_gain_db: -1.0,
    });

    // Standard de-essing.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.2,
        threshold_db: -20.0,
        max_reduction_db: -10.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    });

    // Subtle vibrato: present but controlled, professional feel.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.5,
        depth_cents: 20.0,
        rate_spread_hz: 0.3,
        depth_spread_cents: 8.0,
        onset_sec: 0.18,
    });

    // Moderate Gaussian detuning for natural ensemble spread.
    config.detune = Some(DetuneConfig {
        cents_spread: 10.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Full humanization.
    config.humanize = Some(HumanizeConfig::default());

    // Moderate blending.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.4,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Moderate stereo: professional, not overly wide.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.60));

    // Mastering-grade transparent dynamics.
    config.dynamics = Some(DynamicsPreset::Mastering.to_config());

    // Short room reverb: adds depth without muddying.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.10,
        room_size: 0.25,
        early_reflections: true,
        damping: 0.5,
    });
    config.limiter_enabled = true;

    Ok(config)
}

fn build_live_stage(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Natural EQ: flat with slight low warmth and presence lift.
    config.eq = Some(EqConfig {
        low_freq: 180.0,
        low_gain_db: 0.5,
        mid_freq: 3000.0,
        mid_gain_db: 1.0,
        mid_q: 0.8,
        high_freq: 9000.0,
        high_gain_db: -1.0,
    });

    // Moderate de-essing.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6500.0,
        q: 1.0,
        threshold_db: -20.0,
        max_reduction_db: -8.0,
        attack_sec: 0.001,
        release_sec: 0.055,
    });

    // Natural vibrato: typical singing vibrato rate and depth.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.5,
        depth_cents: 30.0,
        rate_spread_hz: 0.4,
        depth_spread_cents: 10.0,
        onset_sec: 0.20,
    });

    // Moderate Gaussian detuning for natural choir spread.
    config.detune = Some(DetuneConfig {
        cents_spread: 12.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Full humanization.
    config.humanize = Some(HumanizeConfig::default());

    // Moderate blending.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.45,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Moderate-wide stereo for live stage feel.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.75));

    // Broadcast dynamics for consistent live loudness.
    config.dynamics = Some(DynamicsPreset::Broadcast.to_config());

    // Medium room reverb: simulates a concert hall.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.20,
        room_size: 0.50,
        early_reflections: true,
        damping: 0.45,
    });
    config.limiter_enabled = true;

    Ok(config)
}

fn build_intimate(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Very warm EQ: proximity bass boost, mid dip to reduce nasal buildup,
    // aggressive high roll-off for smooth, close texture.
    config.eq = Some(EqConfig {
        low_freq: 200.0,
        low_gain_db: 3.0,
        mid_freq: 2000.0,
        mid_gain_db: -2.0,
        mid_q: 0.7,
        high_freq: 6000.0,
        high_gain_db: -4.0,
    });

    // Gentle de-essing: close mic captures more sibilance.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 5500.0,
        q: 1.0,
        threshold_db: -22.0,
        max_reduction_db: -10.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    });

    // No vibrato: intimate, straight, close sound.
    config.vibrato = None;

    // Minimal detuning: just enough to avoid comb filtering.
    config.detune = Some(DetuneConfig {
        cents_spread: 3.0,
        distribution: DetuneDistribution::Uniform,
        seed: 0,
    });

    // Envelope-only humanization.
    config.humanize = Some(HumanizeConfig {
        enable_breath: false,
        enable_timing: false,
        enable_envelope: true,
        ..HumanizeConfig::default()
    });

    // Light blending.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.25,
        formant_preservation: true,
        harmonic_alignment: false,
        min_period: 30,
        max_period: 300,
    });

    // Very narrow stereo: voices close to center.
    config.stereo = Some(
        StereoChorusConfig::auto_layout(n_voices)?
            .with_stereo_width(0.25)
            .with_mono_compatible(true),
    );

    // Mastering-grade transparent dynamics.
    config.dynamics = Some(DynamicsPreset::Mastering.to_config());

    // No reverb: dry, close proximity sound.
    config.reverb = None;
    config.limiter_enabled = true;

    Ok(config)
}

fn build_epic(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Powerful EQ: low presence for weight, strong mid presence,
    // controlled highs to avoid harshness at high voice counts.
    config.eq = Some(EqConfig {
        low_freq: 150.0,
        low_gain_db: 1.5,
        mid_freq: 2800.0,
        mid_gain_db: 2.0,
        mid_q: 0.8,
        high_freq: 9000.0,
        high_gain_db: -2.0,
    });

    // Aggressive de-essing: many stacked voices amplify sibilance heavily.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 5500.0,
        q: 1.3,
        threshold_db: -16.0,
        max_reduction_db: -14.0,
        attack_sec: 0.001,
        release_sec: 0.040,
    });

    // Deep vibrato: dramatic, cinematic shimmer with maximum spread.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.0,
        depth_cents: 50.0,
        rate_spread_hz: 0.8,
        depth_spread_cents: 20.0,
        onset_sec: 0.20,
    });

    // Wide Gaussian detuning for massive wall-of-sound texture.
    config.detune = Some(DetuneConfig {
        cents_spread: 20.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Full humanization for natural ensemble feel.
    config.humanize = Some(HumanizeConfig::default());

    // Strong blending: fuse voices into one cohesive mass.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.65,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Maximum stereo width.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(1.0));

    // Aggressive dynamics: heavy compression for dense mix.
    config.dynamics = Some(DynamicsPreset::Aggressive.to_config());

    // Large reverb: cinematic spatial depth.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.30,
        room_size: 0.80,
        early_reflections: true,
        damping: 0.55,
    });
    config.limiter_enabled = true;

    Ok(config)
}

fn build_whisper(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Very warm, dark EQ: strong bass presence for proximity,
    // cut mids to reduce nasal quality, heavy high roll-off.
    config.eq = Some(EqConfig {
        low_freq: 250.0,
        low_gain_db: 3.5,
        mid_freq: 2000.0,
        mid_gain_db: -2.5,
        mid_q: 0.6,
        high_freq: 5000.0,
        high_gain_db: -5.0,
    });

    // Light de-essing: whisper has less sibilance energy.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 0.8,
        threshold_db: -24.0,
        max_reduction_db: -6.0,
        attack_sec: 0.001,
        release_sec: 0.060,
    });

    // No vibrato: whispers are straight-toned.
    config.vibrato = None;

    // Minimal detuning: just enough to prevent phasing.
    config.detune = Some(DetuneConfig {
        cents_spread: 2.0,
        distribution: DetuneDistribution::Uniform,
        seed: 0,
    });

    // Envelope-only humanization with slow attack for soft onset.
    config.humanize = Some(HumanizeConfig {
        enable_breath: false,
        enable_timing: false,
        enable_envelope: true,
        envelope: crate::kokoro_chorus_humanize::AmplitudeEnvelope {
            attack_sec: 0.060,
            hold_sec: 0.0,
            decay_sec: 0.040,
            sustain_level: 1.0,
            release_sec: 0.150,
        },
        ..HumanizeConfig::default()
    });

    // Light blending.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.20,
        formant_preservation: true,
        harmonic_alignment: false,
        min_period: 30,
        max_period: 300,
    });

    // Extremely narrow stereo: close, intimate positioning.
    config.stereo = Some(
        StereoChorusConfig::auto_layout(n_voices)?
            .with_stereo_width(0.20)
            .with_mono_compatible(true),
    );

    // Gentle dynamics: don't squash the delicate whisper dynamics.
    config.dynamics = Some(DynamicsPreset::Gentle.to_config());

    // Tiny reverb: just a whisper of air for subtle spatial depth.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.05,
        room_size: 0.15,
        early_reflections: false,
        damping: 0.7,
    });
    config.limiter_enabled = true;

    Ok(config)
}

fn build_choir(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Balanced choral EQ: warm low-mids, presence for intelligibility,
    // smooth highs.
    config.eq = Some(EqConfig {
        low_freq: 180.0,
        low_gain_db: 1.0,
        mid_freq: 3000.0,
        mid_gain_db: 1.0,
        mid_q: 0.8,
        high_freq: 9000.0,
        high_gain_db: -1.5,
    });

    // Moderate de-essing.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6500.0,
        q: 1.2,
        threshold_db: -18.0,
        max_reduction_db: -8.0,
        attack_sec: 0.001,
        release_sec: 0.060,
    });

    // Natural singing vibrato: classic choir shimmer.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.5,
        depth_cents: 35.0,
        rate_spread_hz: 0.5,
        depth_spread_cents: 10.0,
        onset_sec: 0.20,
    });

    // Gaussian detuning: natural choir intonation spread.
    config.detune = Some(DetuneConfig {
        cents_spread: 12.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Full humanization for natural choir feel.
    config.humanize = Some(HumanizeConfig::default());

    // Strong blending with formant preservation and harmonic alignment.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.50,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Wide stereo: spacious choir image.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.80));

    // Gentle dynamics: preserve choir expression.
    config.dynamics = Some(DynamicsPreset::Gentle.to_config());

    // Medium reverb: choral spatial depth.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.20,
        room_size: 0.45,
        early_reflections: true,
        damping: 0.45,
    });
    config.limiter_enabled = true;

    Ok(config)
}

// ---------------------------------------------------------------------------
// Preset validation
// ---------------------------------------------------------------------------

/// Validate that all config values in a [`ChorusMasterConfig`] are within safe
/// operating ranges.
///
/// This performs IEEE 754 finiteness checks on all float fields, range checks
/// on all parameters, and delegates to per-sub-config validators. Returns
/// `Ok(())` if the config is fully valid, or `Err(issues)` with a list of
/// human-readable problem descriptions.
///
/// # Why not just `ChorusMasterConfig::validate()`?
///
/// The existing `validate()` method returns on the first error. This function
/// collects *all* issues for diagnostic reporting.
pub fn validate_preset(config: &ChorusMasterConfig) -> Result<(), Vec<String>> {
    let mut issues = Vec::new();

    // Voice count.
    if config.n_voices == 0 || config.n_voices > 32 {
        issues.push(format!("n_voices = {}: must be in 1..=32", config.n_voices));
    }

    // EQ validation.
    if let Some(ref eq) = config.eq {
        validate_eq(eq, &mut issues);
    }

    // De-esser validation.
    if let Some(ref ds) = config.deesser {
        validate_deesser(ds, &mut issues);
    }

    // Vibrato validation.
    if let Some(ref vib) = config.vibrato {
        validate_vibrato(vib, &mut issues);
    }

    // Detune validation.
    if let Some(ref det) = config.detune {
        validate_detune(det, &mut issues);
    }

    // Humanize validation (sub-config validates internally).
    if let Some(ref hum) = config.humanize {
        if let Err(e) = hum.validate() {
            issues.push(format!("humanize: {e}"));
        }
    }

    // Blend validation.
    if let Some(ref blend) = config.blend {
        if let Err(e) = blend.validate() {
            issues.push(format!("blend: {e}"));
        }
    }

    // Stereo validation.
    if let Some(ref stereo) = config.stereo {
        if let Err(e) = stereo.validate() {
            issues.push(format!("stereo: {e}"));
        }
    }

    // Dynamics validation.
    if let Some(ref dyn_cfg) = config.dynamics {
        if let Err(e) = dyn_cfg.validate() {
            issues.push(format!("dynamics: {e}"));
        }
    }

    // Reverb validation.
    if let Some(ref rev) = config.reverb {
        if let Err(e) = rev.validate() {
            issues.push(format!("reverb: {e}"));
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

fn validate_eq(eq: &EqConfig, issues: &mut Vec<String>) {
    // IEEE 754 finiteness checks.
    for &(name, val) in &[
        ("eq.low_freq", eq.low_freq),
        ("eq.low_gain_db", eq.low_gain_db),
        ("eq.mid_freq", eq.mid_freq),
        ("eq.mid_gain_db", eq.mid_gain_db),
        ("eq.mid_q", eq.mid_q),
        ("eq.high_freq", eq.high_freq),
        ("eq.high_gain_db", eq.high_gain_db),
    ] {
        if !val.is_finite() {
            issues.push(format!("{name} = {val}: not finite"));
        }
    }

    // Range checks (matching EqConfig::validate ranges).
    for &(name, freq) in &[
        ("eq.low_freq", eq.low_freq),
        ("eq.mid_freq", eq.mid_freq),
        ("eq.high_freq", eq.high_freq),
    ] {
        if freq.is_finite() && !(20.0..=20000.0).contains(&freq) {
            issues.push(format!("{name} = {freq}: must be in [20, 20000]"));
        }
    }
    for &(name, gain) in &[
        ("eq.low_gain_db", eq.low_gain_db),
        ("eq.mid_gain_db", eq.mid_gain_db),
        ("eq.high_gain_db", eq.high_gain_db),
    ] {
        if gain.is_finite() && !(-24.0..=24.0).contains(&gain) {
            issues.push(format!("{name} = {gain}: must be in [-24, 24]"));
        }
    }
    if eq.mid_q.is_finite() && (eq.mid_q <= 0.0 || eq.mid_q > 20.0) {
        issues.push(format!("eq.mid_q = {}: must be in (0, 20]", eq.mid_q));
    }
}

fn validate_deesser(ds: &DeEsserConfig, issues: &mut Vec<String>) {
    if !ds.center_freq_hz.is_finite() || ds.center_freq_hz < 100.0 || ds.center_freq_hz > 20000.0 {
        issues.push(format!(
            "deesser.center_freq_hz = {}: must be finite and in [100, 20000]",
            ds.center_freq_hz,
        ));
    }
    if !ds.q.is_finite() || ds.q <= 0.0 || ds.q > 20.0 {
        issues.push(format!(
            "deesser.q = {}: must be finite and in (0, 20]",
            ds.q,
        ));
    }
    if !ds.threshold_db.is_finite() || ds.threshold_db > 0.0 || ds.threshold_db < -96.0 {
        issues.push(format!(
            "deesser.threshold_db = {}: must be finite and in [-96, 0]",
            ds.threshold_db,
        ));
    }
    if !ds.max_reduction_db.is_finite() || ds.max_reduction_db > 0.0 || ds.max_reduction_db < -48.0
    {
        issues.push(format!(
            "deesser.max_reduction_db = {}: must be finite and in [-48, 0]",
            ds.max_reduction_db,
        ));
    }
    if !ds.attack_sec.is_finite() || ds.attack_sec <= 0.0 || ds.attack_sec > 0.1 {
        issues.push(format!(
            "deesser.attack_sec = {}: must be finite and in (0, 0.1]",
            ds.attack_sec,
        ));
    }
    if !ds.release_sec.is_finite() || ds.release_sec <= 0.0 || ds.release_sec > 1.0 {
        issues.push(format!(
            "deesser.release_sec = {}: must be finite and in (0, 1]",
            ds.release_sec,
        ));
    }
}

fn validate_vibrato(vib: &VibratoConfig, issues: &mut Vec<String>) {
    if !vib.rate_hz.is_finite() || vib.rate_hz < 0.5 || vib.rate_hz > 20.0 {
        issues.push(format!(
            "vibrato.rate_hz = {}: must be finite and in [0.5, 20]",
            vib.rate_hz,
        ));
    }
    if !vib.depth_cents.is_finite() || vib.depth_cents < 0.0 || vib.depth_cents > 200.0 {
        issues.push(format!(
            "vibrato.depth_cents = {}: must be finite and in [0, 200]",
            vib.depth_cents,
        ));
    }
    if !vib.rate_spread_hz.is_finite() || vib.rate_spread_hz < 0.0 || vib.rate_spread_hz > 5.0 {
        issues.push(format!(
            "vibrato.rate_spread_hz = {}: must be finite and in [0, 5]",
            vib.rate_spread_hz,
        ));
    }
    if !vib.depth_spread_cents.is_finite()
        || vib.depth_spread_cents < 0.0
        || vib.depth_spread_cents > 100.0
    {
        issues.push(format!(
            "vibrato.depth_spread_cents = {}: must be finite and in [0, 100]",
            vib.depth_spread_cents,
        ));
    }
    if !vib.onset_sec.is_finite() || vib.onset_sec < 0.0 || vib.onset_sec > 5.0 {
        issues.push(format!(
            "vibrato.onset_sec = {}: must be finite and in [0, 5]",
            vib.onset_sec,
        ));
    }
}

fn validate_detune(det: &DetuneConfig, issues: &mut Vec<String>) {
    if !det.cents_spread.is_finite() || det.cents_spread < 0.0 || det.cents_spread > 50.0 {
        issues.push(format!(
            "detune.cents_spread = {}: must be finite and in [0, 50]",
            det.cents_spread,
        ));
    }
}

// ---------------------------------------------------------------------------
// Preset comparison
// ---------------------------------------------------------------------------

/// A single difference between two configs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConfigDifference {
    /// Which parameter differs (e.g. "eq.low_gain_db", "reverb", "stereo_width").
    pub field: String,
    /// Description of the left (first) config's value.
    pub left: String,
    /// Description of the right (second) config's value.
    pub right: String,
}

impl ConfigDifference {
    fn new(field: impl Into<String>, left: impl Into<String>, right: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            left: left.into(),
            right: right.into(),
        }
    }
}

/// Comparison result between two [`ChorusMasterConfig`] instances.
///
/// Collects all differences between two configs and provides
/// human-readable formatting.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PresetComparison {
    /// Name/label for the first config.
    pub left_name: String,
    /// Name/label for the second config.
    pub right_name: String,
    /// All differences found between the two configs.
    pub differences: Vec<ConfigDifference>,
}

impl PresetComparison {
    /// Compare two configs with the given names.
    #[must_use]
    pub fn compare(
        left_name: impl Into<String>,
        left: &ChorusMasterConfig,
        right_name: impl Into<String>,
        right: &ChorusMasterConfig,
    ) -> Self {
        let mut diffs = Vec::new();
        let left_name = left_name.into();
        let right_name = right_name.into();

        // Voice count.
        if left.n_voices != right.n_voices {
            diffs.push(ConfigDifference::new(
                "n_voices",
                left.n_voices.to_string(),
                right.n_voices.to_string(),
            ));
        }

        // EQ.
        compare_optional_eq(&left.eq, &right.eq, &mut diffs);

        // De-esser.
        compare_optional_deesser(&left.deesser, &right.deesser, &mut diffs);

        // Vibrato.
        compare_optional_vibrato(&left.vibrato, &right.vibrato, &mut diffs);

        // Detune.
        compare_optional_detune(&left.detune, &right.detune, &mut diffs);

        // Humanize (presence only — detailed internal comparison is complex).
        match (&left.humanize, &right.humanize) {
            (Some(l), Some(r)) => {
                if l.enable_breath != r.enable_breath {
                    diffs.push(ConfigDifference::new(
                        "humanize.enable_breath",
                        l.enable_breath.to_string(),
                        r.enable_breath.to_string(),
                    ));
                }
                if l.enable_timing != r.enable_timing {
                    diffs.push(ConfigDifference::new(
                        "humanize.enable_timing",
                        l.enable_timing.to_string(),
                        r.enable_timing.to_string(),
                    ));
                }
                if l.enable_envelope != r.enable_envelope {
                    diffs.push(ConfigDifference::new(
                        "humanize.enable_envelope",
                        l.enable_envelope.to_string(),
                        r.enable_envelope.to_string(),
                    ));
                }
            }
            (None, Some(_)) => {
                diffs.push(ConfigDifference::new("humanize", "None", "Some(...)"));
            }
            (Some(_), None) => {
                diffs.push(ConfigDifference::new("humanize", "Some(...)", "None"));
            }
            (None, None) => {}
        }

        // Blend.
        compare_optional_blend(&left.blend, &right.blend, &mut diffs);

        // Stereo.
        compare_optional_stereo(&left.stereo, &right.stereo, &mut diffs);

        // Dynamics (presence only — internal config is multi-band).
        match (&left.dynamics, &right.dynamics) {
            (None, Some(_)) => {
                diffs.push(ConfigDifference::new("dynamics", "None", "Some(...)"));
            }
            (Some(_), None) => {
                diffs.push(ConfigDifference::new("dynamics", "Some(...)", "None"));
            }
            _ => {}
        }

        // Reverb.
        compare_optional_reverb(&left.reverb, &right.reverb, &mut diffs);

        // Limiter.
        if left.limiter_enabled != right.limiter_enabled {
            diffs.push(ConfigDifference::new(
                "limiter_enabled",
                left.limiter_enabled.to_string(),
                right.limiter_enabled.to_string(),
            ));
        }

        Self {
            left_name,
            right_name,
            differences: diffs,
        }
    }

    /// Whether the two configs are identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.differences.is_empty()
    }

    /// Format the comparison as a human-readable string.
    #[must_use]
    pub fn format_comparison(&self) -> String {
        if self.differences.is_empty() {
            return format!(
                "Presets '{}' and '{}' are identical.",
                self.left_name, self.right_name,
            );
        }

        let mut out = format!(
            "Comparison: '{}' vs '{}' ({} difference{}):\n",
            self.left_name,
            self.right_name,
            self.differences.len(),
            if self.differences.len() == 1 { "" } else { "s" },
        );

        for diff in &self.differences {
            out.push_str(&format!(
                "  {:<30} {:>15}  ->  {:<15}\n",
                diff.field, diff.left, diff.right,
            ));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Compare two floats with a small epsilon for formatting differences.
fn floats_differ(a: f32, b: f32) -> bool {
    // IEEE 754 safety: if either is non-finite, compare as bits.
    if !a.is_finite() || !b.is_finite() {
        return a.to_bits() != b.to_bits();
    }
    (a - b).abs() > 1e-6
}

fn compare_optional_eq(
    left: &Option<EqConfig>,
    right: &Option<EqConfig>,
    diffs: &mut Vec<ConfigDifference>,
) {
    match (left, right) {
        (Some(l), Some(r)) => {
            if floats_differ(l.low_freq, r.low_freq) {
                diffs.push(ConfigDifference::new(
                    "eq.low_freq",
                    format!("{:.1}", l.low_freq),
                    format!("{:.1}", r.low_freq),
                ));
            }
            if floats_differ(l.low_gain_db, r.low_gain_db) {
                diffs.push(ConfigDifference::new(
                    "eq.low_gain_db",
                    format!("{:.1}", l.low_gain_db),
                    format!("{:.1}", r.low_gain_db),
                ));
            }
            if floats_differ(l.mid_freq, r.mid_freq) {
                diffs.push(ConfigDifference::new(
                    "eq.mid_freq",
                    format!("{:.1}", l.mid_freq),
                    format!("{:.1}", r.mid_freq),
                ));
            }
            if floats_differ(l.mid_gain_db, r.mid_gain_db) {
                diffs.push(ConfigDifference::new(
                    "eq.mid_gain_db",
                    format!("{:.1}", l.mid_gain_db),
                    format!("{:.1}", r.mid_gain_db),
                ));
            }
            if floats_differ(l.mid_q, r.mid_q) {
                diffs.push(ConfigDifference::new(
                    "eq.mid_q",
                    format!("{:.2}", l.mid_q),
                    format!("{:.2}", r.mid_q),
                ));
            }
            if floats_differ(l.high_freq, r.high_freq) {
                diffs.push(ConfigDifference::new(
                    "eq.high_freq",
                    format!("{:.1}", l.high_freq),
                    format!("{:.1}", r.high_freq),
                ));
            }
            if floats_differ(l.high_gain_db, r.high_gain_db) {
                diffs.push(ConfigDifference::new(
                    "eq.high_gain_db",
                    format!("{:.1}", l.high_gain_db),
                    format!("{:.1}", r.high_gain_db),
                ));
            }
        }
        (None, Some(_)) => {
            diffs.push(ConfigDifference::new("eq", "None", "Some(...)"));
        }
        (Some(_), None) => {
            diffs.push(ConfigDifference::new("eq", "Some(...)", "None"));
        }
        (None, None) => {}
    }
}

fn compare_optional_deesser(
    left: &Option<DeEsserConfig>,
    right: &Option<DeEsserConfig>,
    diffs: &mut Vec<ConfigDifference>,
) {
    match (left, right) {
        (Some(l), Some(r)) => {
            if floats_differ(l.center_freq_hz, r.center_freq_hz) {
                diffs.push(ConfigDifference::new(
                    "deesser.center_freq_hz",
                    format!("{:.1}", l.center_freq_hz),
                    format!("{:.1}", r.center_freq_hz),
                ));
            }
            if floats_differ(l.q, r.q) {
                diffs.push(ConfigDifference::new(
                    "deesser.q",
                    format!("{:.2}", l.q),
                    format!("{:.2}", r.q),
                ));
            }
            if floats_differ(l.threshold_db, r.threshold_db) {
                diffs.push(ConfigDifference::new(
                    "deesser.threshold_db",
                    format!("{:.1}", l.threshold_db),
                    format!("{:.1}", r.threshold_db),
                ));
            }
            if floats_differ(l.max_reduction_db, r.max_reduction_db) {
                diffs.push(ConfigDifference::new(
                    "deesser.max_reduction_db",
                    format!("{:.1}", l.max_reduction_db),
                    format!("{:.1}", r.max_reduction_db),
                ));
            }
        }
        (None, Some(_)) => {
            diffs.push(ConfigDifference::new("deesser", "None", "Some(...)"));
        }
        (Some(_), None) => {
            diffs.push(ConfigDifference::new("deesser", "Some(...)", "None"));
        }
        (None, None) => {}
    }
}

fn compare_optional_vibrato(
    left: &Option<VibratoConfig>,
    right: &Option<VibratoConfig>,
    diffs: &mut Vec<ConfigDifference>,
) {
    match (left, right) {
        (Some(l), Some(r)) => {
            if floats_differ(l.rate_hz, r.rate_hz) {
                diffs.push(ConfigDifference::new(
                    "vibrato.rate_hz",
                    format!("{:.1}", l.rate_hz),
                    format!("{:.1}", r.rate_hz),
                ));
            }
            if floats_differ(l.depth_cents, r.depth_cents) {
                diffs.push(ConfigDifference::new(
                    "vibrato.depth_cents",
                    format!("{:.1}", l.depth_cents),
                    format!("{:.1}", r.depth_cents),
                ));
            }
            if floats_differ(l.rate_spread_hz, r.rate_spread_hz) {
                diffs.push(ConfigDifference::new(
                    "vibrato.rate_spread_hz",
                    format!("{:.2}", l.rate_spread_hz),
                    format!("{:.2}", r.rate_spread_hz),
                ));
            }
            if floats_differ(l.depth_spread_cents, r.depth_spread_cents) {
                diffs.push(ConfigDifference::new(
                    "vibrato.depth_spread_cents",
                    format!("{:.1}", l.depth_spread_cents),
                    format!("{:.1}", r.depth_spread_cents),
                ));
            }
        }
        (None, Some(_)) => {
            diffs.push(ConfigDifference::new("vibrato", "None", "Some(...)"));
        }
        (Some(_), None) => {
            diffs.push(ConfigDifference::new("vibrato", "Some(...)", "None"));
        }
        (None, None) => {}
    }
}

fn compare_optional_detune(
    left: &Option<DetuneConfig>,
    right: &Option<DetuneConfig>,
    diffs: &mut Vec<ConfigDifference>,
) {
    match (left, right) {
        (Some(l), Some(r)) => {
            if floats_differ(l.cents_spread, r.cents_spread) {
                diffs.push(ConfigDifference::new(
                    "detune.cents_spread",
                    format!("{:.1}", l.cents_spread),
                    format!("{:.1}", r.cents_spread),
                ));
            }
        }
        (None, Some(_)) => {
            diffs.push(ConfigDifference::new("detune", "None", "Some(...)"));
        }
        (Some(_), None) => {
            diffs.push(ConfigDifference::new("detune", "Some(...)", "None"));
        }
        (None, None) => {}
    }
}

fn compare_optional_blend(
    left: &Option<EnsembleBlendConfig>,
    right: &Option<EnsembleBlendConfig>,
    diffs: &mut Vec<ConfigDifference>,
) {
    match (left, right) {
        (Some(l), Some(r)) => {
            if floats_differ(l.blend_strength, r.blend_strength) {
                diffs.push(ConfigDifference::new(
                    "blend.blend_strength",
                    format!("{:.2}", l.blend_strength),
                    format!("{:.2}", r.blend_strength),
                ));
            }
            if l.formant_preservation != r.formant_preservation {
                diffs.push(ConfigDifference::new(
                    "blend.formant_preservation",
                    l.formant_preservation.to_string(),
                    r.formant_preservation.to_string(),
                ));
            }
            if l.harmonic_alignment != r.harmonic_alignment {
                diffs.push(ConfigDifference::new(
                    "blend.harmonic_alignment",
                    l.harmonic_alignment.to_string(),
                    r.harmonic_alignment.to_string(),
                ));
            }
        }
        (None, Some(_)) => {
            diffs.push(ConfigDifference::new("blend", "None", "Some(...)"));
        }
        (Some(_), None) => {
            diffs.push(ConfigDifference::new("blend", "Some(...)", "None"));
        }
        (None, None) => {}
    }
}

fn compare_optional_stereo(
    left: &Option<StereoChorusConfig>,
    right: &Option<StereoChorusConfig>,
    diffs: &mut Vec<ConfigDifference>,
) {
    match (left, right) {
        (Some(l), Some(r)) => {
            if floats_differ(l.stereo_width, r.stereo_width) {
                diffs.push(ConfigDifference::new(
                    "stereo.stereo_width",
                    format!("{:.2}", l.stereo_width),
                    format!("{:.2}", r.stereo_width),
                ));
            }
            if l.mono_compatible != r.mono_compatible {
                diffs.push(ConfigDifference::new(
                    "stereo.mono_compatible",
                    l.mono_compatible.to_string(),
                    r.mono_compatible.to_string(),
                ));
            }
            if l.positions.len() != r.positions.len() {
                diffs.push(ConfigDifference::new(
                    "stereo.positions.len",
                    l.positions.len().to_string(),
                    r.positions.len().to_string(),
                ));
            }
        }
        (None, Some(_)) => {
            diffs.push(ConfigDifference::new("stereo", "None", "Some(...)"));
        }
        (Some(_), None) => {
            diffs.push(ConfigDifference::new("stereo", "Some(...)", "None"));
        }
        (None, None) => {}
    }
}

fn compare_optional_reverb(
    left: &Option<ReverbConfig>,
    right: &Option<ReverbConfig>,
    diffs: &mut Vec<ConfigDifference>,
) {
    match (left, right) {
        (Some(l), Some(r)) => {
            if floats_differ(l.reverb_mix, r.reverb_mix) {
                diffs.push(ConfigDifference::new(
                    "reverb.reverb_mix",
                    format!("{:.2}", l.reverb_mix),
                    format!("{:.2}", r.reverb_mix),
                ));
            }
            if floats_differ(l.room_size, r.room_size) {
                diffs.push(ConfigDifference::new(
                    "reverb.room_size",
                    format!("{:.2}", l.room_size),
                    format!("{:.2}", r.room_size),
                ));
            }
            if l.early_reflections != r.early_reflections {
                diffs.push(ConfigDifference::new(
                    "reverb.early_reflections",
                    l.early_reflections.to_string(),
                    r.early_reflections.to_string(),
                ));
            }
            if floats_differ(l.damping, r.damping) {
                diffs.push(ConfigDifference::new(
                    "reverb.damping",
                    format!("{:.2}", l.damping),
                    format!("{:.2}", r.damping),
                ));
            }
        }
        (None, Some(_)) => {
            diffs.push(ConfigDifference::new("reverb", "None", "Some(...)"));
        }
        (Some(_), None) => {
            diffs.push(ConfigDifference::new("reverb", "Some(...)", "None"));
        }
        (None, None) => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets_build_4_voices() {
        for &preset in ChorusPreset::ALL {
            let config = preset
                .to_config(4)
                .unwrap_or_else(|_| panic!("preset {preset:?} should build with 4 voices"));
            validate_preset(&config)
                .unwrap_or_else(|_| panic!("preset {preset:?} should pass validation"));
        }
    }

    #[test]
    fn test_all_presets_build_1_voice() {
        for &preset in ChorusPreset::ALL {
            let config = preset
                .to_config(1)
                .unwrap_or_else(|_| panic!("preset {preset:?} should build with 1 voice"));
            validate_preset(&config).unwrap_or_else(|_| panic!("preset {preset:?} should pass validation with 1 voice"));
        }
    }

    #[test]
    fn test_all_presets_build_32_voices() {
        for &preset in ChorusPreset::ALL {
            let config = preset
                .to_config(32)
                .unwrap_or_else(|_| panic!("preset {preset:?} should build with 32 voices"));
            validate_preset(&config).unwrap_or_else(|_| panic!("preset {preset:?} should pass validation with 32 voices"));
        }
    }

    #[test]
    fn test_preset_zero_voices_fails() {
        for &preset in ChorusPreset::ALL {
            assert!(
                preset.to_config(0).is_err(),
                "preset {preset:?} should fail with 0 voices",
            );
        }
    }

    #[test]
    fn test_preset_33_voices_fails() {
        for &preset in ChorusPreset::ALL {
            assert!(
                preset.to_config(33).is_err(),
                "preset {preset:?} should fail with 33 voices",
            );
        }
    }

    #[test]
    fn test_preset_names_are_unique() {
        let names: Vec<&str> = ChorusPreset::ALL.iter().map(|p| p.name()).collect();
        for (i, a) in names.iter().enumerate() {
            for (j, b) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "preset names must be unique");
                }
            }
        }
    }

    #[test]
    fn test_preset_descriptions_non_empty() {
        for &preset in ChorusPreset::ALL {
            assert!(
                !preset.description().is_empty(),
                "preset {preset:?} description must not be empty",
            );
        }
    }

    #[test]
    fn test_comparison_identical_presets() {
        let a = ChorusPreset::Studio.to_config(4).unwrap();
        let b = ChorusPreset::Studio.to_config(4).unwrap();
        let cmp = PresetComparison::compare("Studio A", &a, "Studio B", &b);
        assert!(
            cmp.is_identical(),
            "same preset should produce identical configs: {}",
            cmp.format_comparison(),
        );
    }

    #[test]
    fn test_comparison_different_presets() {
        let a = ChorusPreset::Intimate.to_config(4).unwrap();
        let b = ChorusPreset::Cathedral.to_config(4).unwrap();
        let cmp = PresetComparison::compare("Intimate", &a, "Cathedral", &b);
        assert!(!cmp.is_identical(), "different presets should differ");
        // Should have multiple differences.
        assert!(
            cmp.differences.len() > 5,
            "Intimate vs Cathedral should have many differences, got {}",
            cmp.differences.len(),
        );
    }

    #[test]
    fn test_format_comparison_output() {
        let a = ChorusPreset::Whisper.to_config(4).unwrap();
        let b = ChorusPreset::Epic.to_config(4).unwrap();
        let cmp = PresetComparison::compare("Whisper", &a, "Epic", &b);
        let output = cmp.format_comparison();
        assert!(output.contains("Whisper"));
        assert!(output.contains("Epic"));
        assert!(output.contains("difference"));
    }

    #[test]
    fn test_validate_preset_catches_bad_config() {
        let mut config = ChorusMasterConfig::new(4).unwrap();
        config.eq = Some(EqConfig {
            low_freq: f32::NAN,
            low_gain_db: 0.0,
            mid_freq: 1500.0,
            mid_gain_db: 0.0,
            mid_q: 1.0,
            high_freq: 8000.0,
            high_gain_db: 0.0,
        });
        let result = validate_preset(&config);
        assert!(result.is_err());
        let issues = result.unwrap_err();
        assert!(
            issues.iter().any(|s| s.contains("low_freq")),
            "should report NaN in low_freq: {issues:?}",
        );
    }

    #[test]
    fn test_all_presets_pass_master_validate() {
        for &preset in ChorusPreset::ALL {
            let config = preset.to_config(8).unwrap();
            config.validate().unwrap_or_else(|_| panic!("preset {preset:?} should pass ChorusMasterConfig::validate()"));
        }
    }

    #[test]
    fn test_acappella_has_no_reverb() {
        let config = ChorusPreset::ACapella.to_config(4).unwrap();
        assert!(config.reverb.is_none(), "ACapella should have no reverb");
    }

    #[test]
    fn test_cathedral_has_large_reverb() {
        let config = ChorusPreset::Cathedral.to_config(4).unwrap();
        let reverb = config
            .reverb
            .as_ref()
            .expect("Cathedral should have reverb");
        assert!(
            reverb.room_size >= 0.8,
            "Cathedral room_size should be >= 0.8, got {}",
            reverb.room_size,
        );
        assert!(
            reverb.reverb_mix >= 0.30,
            "Cathedral reverb_mix should be >= 0.30, got {}",
            reverb.reverb_mix,
        );
    }

    #[test]
    fn test_intimate_has_narrow_stereo() {
        let config = ChorusPreset::Intimate.to_config(4).unwrap();
        let stereo = config.stereo.as_ref().expect("Intimate should have stereo");
        assert!(
            stereo.stereo_width <= 0.30,
            "Intimate stereo_width should be <= 0.30, got {}",
            stereo.stereo_width,
        );
        assert!(stereo.mono_compatible, "Intimate should be mono-compatible");
    }

    #[test]
    fn test_whisper_has_no_vibrato() {
        let config = ChorusPreset::Whisper.to_config(4).unwrap();
        assert!(config.vibrato.is_none(), "Whisper should have no vibrato");
    }

    #[test]
    fn test_epic_has_wide_detuning() {
        let config = ChorusPreset::Epic.to_config(4).unwrap();
        let detune = config.detune.as_ref().expect("Epic should have detune");
        assert!(
            detune.cents_spread >= 18.0,
            "Epic cents_spread should be >= 18, got {}",
            detune.cents_spread,
        );
    }

    #[test]
    fn test_choir_has_gaussian_detuning() {
        let config = ChorusPreset::Choir.to_config(4).unwrap();
        let detune = config.detune.as_ref().expect("Choir should have detune");
        assert!(
            matches!(detune.distribution, DetuneDistribution::Gaussian),
            "Choir should use Gaussian detuning",
        );
    }
}

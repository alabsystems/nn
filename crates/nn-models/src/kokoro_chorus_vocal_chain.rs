// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production vocal chain presets combining all chorus processing modules.
//!
//! Each [`VocalChainPreset`] configures a complete [`ChorusMasterConfig`]
//! tuned for a specific vocal production style. Presets use the full range
//! of available processing stages: alignment, EQ, de-essing, vibrato,
//! detune, formant preservation, humanize, breath, transient, bleed,
//! blend, stereo, spatial, width, dynamics, saturation, reverb,
//! convolution reverb, and the bus limiter.
//!
//! The [`validate_chain`] function checks for problematic module
//! combinations (e.g., reverb + convolution muddiness, saturation +
//! high width phase issues) and returns warnings with severity levels.
//!
//! # Presets
//!
//! | Preset             | Character                                           | Modules |
//! |--------------------|-----------------------------------------------------|---------|
//! | `PopChoir`         | Bright, wide, compressed, subtle pitch + doubling   | 16      |
//! | `GospelChoir`      | Warm, dynamic, strong vibrato and breath             | 14      |
//! | `AmbientPad`       | Ethereal, wide stereo, heavy reverb, gentle sustain | 14      |
//! | `BarbershopQuartet`| Tight harmony, minimal effects, close-mic            | 9       |
//! | `OperaChorus`      | Full dynamics, cathedral spatial, formant preserve   | 15      |
//! | `ElectronicVocals` | Heavy pitch correction, exciter, saturation, width  | 13      |
//! | `Whisper`          | Very quiet, close-mic, lots of breath, minimal      | 10      |
//! | `Announcer`        | Broadcast-quality, heavy dynamics, de-essed, staged | 12      |
//!
//! Part of #4264, #3351.

use crate::kokoro_chorus_alignment::AlignmentConfig;
use crate::kokoro_chorus_bleed::BleedConfig;
use crate::kokoro_chorus_blend::EnsembleBlendConfig;
use crate::kokoro_chorus_breath::BreathConfig;
use crate::kokoro_chorus_convolution::ConvolutionConfig;
use crate::kokoro_chorus_detune::{DetuneConfig, DetuneDistribution};
use crate::kokoro_chorus_dynamics::DynamicsPreset;
use crate::kokoro_chorus_eq::{DeEsserConfig, EqConfig};
use crate::kokoro_chorus_formant::FormantPreserveConfig;
use crate::kokoro_chorus_humanize::HumanizeConfig;
use crate::kokoro_chorus_pipeline::ChorusMasterConfig;
use crate::kokoro_chorus_reverb::ReverbConfig;
use crate::kokoro_chorus_saturation::{SaturationConfig, SaturationMode};
use crate::kokoro_chorus_spatial::SpatialConfig;
use crate::kokoro_chorus_stereo::StereoChorusConfig;
use crate::kokoro_chorus_transient::TransientConfig;
use crate::kokoro_chorus_vibrato::VibratoConfig;
use crate::kokoro_chorus_width::StereoWidthConfig;
use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// VocalChainPreset enum
// ---------------------------------------------------------------------------

/// Production vocal chain presets, each combining multiple chorus modules
/// into a complete signal chain tuned for a specific use case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VocalChainPreset {
    /// Bright, wide, compressed pop choir with subtle pitch correction feel
    /// and vocal doubling character. Modules: alignment, EQ, de-esser,
    /// vibrato, detune, formant, humanize, breath, transient, blend, stereo,
    /// width, dynamics, saturation, reverb, limiter.
    PopChoir,

    /// Warm, dynamic gospel choir with strong vibrato and prominent breath.
    /// Modules: alignment, EQ, de-esser, vibrato, detune, formant, humanize,
    /// breath, bleed, blend, stereo, spatial, dynamics, limiter.
    GospelChoir,

    /// Ethereal ambient vocal pad with wide stereo, heavy reverb, and gentle
    /// freeze-like sustain. Modules: EQ, vibrato, detune, formant, humanize,
    /// blend, stereo, width, dynamics, saturation, reverb, convolution,
    /// spatial, limiter.
    AmbientPad,

    /// Tight barbershop harmony with minimal effects and close-mic character.
    /// Modules: alignment, EQ, de-esser, detune, humanize, blend, stereo,
    /// dynamics, limiter.
    BarbershopQuartet,

    /// Full-dynamics opera chorus with cathedral spatial and formant
    /// preservation. Modules: alignment, EQ, de-esser, vibrato, detune,
    /// formant, humanize, breath, bleed, blend, stereo, spatial, convolution,
    /// dynamics, limiter.
    OperaChorus,

    /// Electronic vocals with heavy pitch correction, exciter-like brightness,
    /// saturation, and exaggerated stereo width. Modules: alignment, EQ,
    /// de-esser, detune, transient, blend, stereo, width, dynamics,
    /// saturation, reverb, convolution, limiter.
    ElectronicVocals,

    /// Very quiet, close-mic whisper with lots of breath and minimal dynamics.
    /// Modules: EQ, de-esser, detune, humanize, breath, blend, stereo,
    /// width, dynamics, limiter.
    Whisper,

    /// Broadcast-quality announcer voice with heavy dynamics, aggressive
    /// de-essing, and precise gain staging. Modules: alignment, EQ, de-esser,
    /// detune, humanize, transient, blend, stereo, width, dynamics,
    /// saturation, limiter.
    Announcer,
}

impl VocalChainPreset {
    /// All available vocal chain presets, in definition order.
    pub const ALL: &'static [Self] = &[
        Self::PopChoir,
        Self::GospelChoir,
        Self::AmbientPad,
        Self::BarbershopQuartet,
        Self::OperaChorus,
        Self::ElectronicVocals,
        Self::Whisper,
        Self::Announcer,
    ];

    /// Human-readable name for this preset.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::PopChoir => "Pop Choir",
            Self::GospelChoir => "Gospel Choir",
            Self::AmbientPad => "Ambient Pad",
            Self::BarbershopQuartet => "Barbershop Quartet",
            Self::OperaChorus => "Opera Chorus",
            Self::ElectronicVocals => "Electronic Vocals",
            Self::Whisper => "Whisper",
            Self::Announcer => "Announcer",
        }
    }

    /// Short description of the preset's sonic character.
    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::PopChoir => "Bright, wide, compressed, with subtle pitch correction and doubling",
            Self::GospelChoir => "Warm, dynamic, with strong vibrato and breath",
            Self::AmbientPad => "Ethereal, wide stereo, heavy reverb, gentle freeze-like sustain",
            Self::BarbershopQuartet => "Tight harmony, minimal effects, close-mic character",
            Self::OperaChorus => "Full dynamics, cathedral spatial, formant preservation",
            Self::ElectronicVocals => "Heavy pitch correction, bright, saturated, wide stereo",
            Self::Whisper => "Very quiet, close-mic, lots of breath, minimal dynamics",
            Self::Announcer => "Broadcast-quality, heavy dynamics, de-essed, gain-staged",
        }
    }

    /// Count of active processing modules for this preset.
    ///
    /// Counts each `Some(...)` field plus `limiter_enabled` in the generated
    /// config. Useful for diagnostics and testing.
    #[must_use]
    pub fn module_count(self) -> usize {
        // Build the config with 4 voices (arbitrary valid count) and count.
        let config = self
            .to_config(4)
            .expect("preset should build with 4 voices");
        count_active_modules(&config)
    }
}

/// Count active processing modules in a config.
fn count_active_modules(config: &ChorusMasterConfig) -> usize {
    let mut count = 0;
    if config.alignment.is_some() {
        count += 1;
    }
    if config.eq.is_some() {
        count += 1;
    }
    if config.deesser.is_some() {
        count += 1;
    }
    if config.vibrato.is_some() {
        count += 1;
    }
    if config.detune.is_some() {
        count += 1;
    }
    if config.humanize.is_some() {
        count += 1;
    }
    if config.blend.is_some() {
        count += 1;
    }
    if config.stereo.is_some() {
        count += 1;
    }
    if config.dynamics.is_some() {
        count += 1;
    }
    if config.saturation.is_some() {
        count += 1;
    }
    if config.reverb.is_some() {
        count += 1;
    }
    if config.breath.is_some() {
        count += 1;
    }
    if config.spatial.is_some() {
        count += 1;
    }
    if config.transient.is_some() {
        count += 1;
    }
    if config.bleed.is_some() {
        count += 1;
    }
    if config.width.is_some() {
        count += 1;
    }
    if config.convolution.is_some() {
        count += 1;
    }
    if config.limiter_enabled {
        count += 1;
    }
    if config.formant_preserve.is_some() {
        count += 1;
    }
    count
}

// ---------------------------------------------------------------------------
// VocalChain builder
// ---------------------------------------------------------------------------

/// Builder for complete vocal chain configurations.
///
/// Converts a [`VocalChainPreset`] into a fully-configured
/// [`ChorusMasterConfig`] using the pipeline's available sub-configs.
pub struct VocalChain;

impl VocalChain {
    /// Build a complete [`ChorusMasterConfig`] from a preset.
    ///
    /// # Errors
    ///
    /// Returns [`KokoroError::InvalidConfig`] if `n_voices` is 0 or > 32.
    pub fn from_preset(
        preset: VocalChainPreset,
        n_voices: usize,
    ) -> Result<ChorusMasterConfig, KokoroError> {
        match preset {
            VocalChainPreset::PopChoir => build_pop_choir(n_voices),
            VocalChainPreset::GospelChoir => build_gospel_choir(n_voices),
            VocalChainPreset::AmbientPad => build_ambient_pad(n_voices),
            VocalChainPreset::BarbershopQuartet => build_barbershop_quartet(n_voices),
            VocalChainPreset::OperaChorus => build_opera_chorus(n_voices),
            VocalChainPreset::ElectronicVocals => build_electronic_vocals(n_voices),
            VocalChainPreset::Whisper => build_whisper(n_voices),
            VocalChainPreset::Announcer => build_announcer(n_voices),
        }
    }
}

impl VocalChainPreset {
    /// Convenience: build config directly from the preset variant.
    pub fn to_config(self, n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
        VocalChain::from_preset(self, n_voices)
    }
}

// ---------------------------------------------------------------------------
// Preset builders
// ---------------------------------------------------------------------------

/// Pop Choir: bright, wide, compressed, with subtle pitch correction feel
/// and vocal doubling character. 16 modules active.
///
/// Signal chain rationale:
/// - Tight alignment keeps doubled voices phase-coherent
/// - Bright EQ (3.5 kHz presence, air at 10 kHz) for radio-ready pop sound
/// - Aggressive de-essing (stacked voices amplify sibilance)
/// - Moderate vibrato with tight spread for polished shimmer
/// - Gaussian detuning for natural ensemble width
/// - Formant preservation prevents chipmunk on wider voices
/// - Transient boost sharpens consonant attacks for pop clarity
/// - Moderate breath for realism without distraction
/// - Wide stereo (0.85) with mid/side enhancement
/// - Broadcast dynamics for consistent loudness
/// - Tape saturation for analog glue
/// - Short/medium reverb for spatial depth without muddiness
/// - Limiter prevents overs
fn build_pop_choir(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Tight alignment for polished pop ensemble.
    config.alignment = Some(AlignmentConfig::new(0.75)?);

    // Bright pop EQ: low warmth, strong presence boost at 3.5 kHz,
    // slight air shelf for sparkle.
    config.eq = Some(EqConfig {
        low_freq: 160.0,
        low_gain_db: 1.0,
        mid_freq: 3500.0,
        mid_gain_db: 2.5,
        mid_q: 0.9,
        high_freq: 10000.0,
        high_gain_db: 0.5,
    });

    // Aggressive de-essing: stacked pop voices create harsh sibilance.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 5500.0,
        q: 1.4,
        threshold_db: -20.0,
        max_reduction_db: -12.0,
        attack_sec: 0.001,
        release_sec: 0.040,
    });

    // Polished vibrato: moderate rate with tight spread for commercial sound.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.8,
        depth_cents: 25.0,
        rate_spread_hz: 0.3,
        depth_spread_cents: 8.0,
        onset_sec: 0.15,
    });

    // Gaussian detuning for natural ensemble width.
    config.detune = Some(DetuneConfig {
        cents_spread: 10.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Formant preservation: prevent chipmunk on wider detuned voices.
    config.formant_preserve = Some(FormantPreserveConfig::default());

    // Full humanization for natural feel.
    config.humanize = Some(HumanizeConfig::default());

    // Moderate breath: present but not distracting.
    config.breath = Some(
        BreathConfig::new()
            .with_noise_level(0.02)
            .with_duration_ms(100.0)
            .with_stagger_ms(30.0),
    );

    // Transient boost: sharpen consonant attacks for pop clarity.
    config.transient = Some(
        TransientConfig::new()
            .with_attack(3.0)
            .with_sustain(-0.5)
            .with_sensitivity(2.0),
    );

    // Moderate ensemble blending with harmonic alignment.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.45,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Wide stereo imaging for spacious pop sound.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.85));

    // Mid/side width enhancement with bass mono.
    config.width = Some(
        StereoWidthConfig::new()
            .with_width(1.3)
            .with_bass_mono_freq(100.0),
    );

    // Broadcast dynamics for consistent loudness.
    config.dynamics = Some(DynamicsPreset::Broadcast.to_config());

    // Tape saturation: analog glue binding voices together.
    config.saturation = Some(
        SaturationConfig::new()
            .with_drive(0.18)
            .with_mix(0.45)
            .with_mode(SaturationMode::Tape)
            .with_output_gain_db(-0.5),
    );

    // Medium reverb: spatial depth without muddiness.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.15,
        room_size: 0.35,
        early_reflections: true,
        damping: 0.50,
    });

    config.limiter_enabled = true;
    Ok(config)
}

/// Gospel Choir: warm, dynamic, strong vibrato with prominent breath.
/// 14 modules active.
///
/// Signal chain rationale:
/// - Moderate alignment preserves natural gospel timing feel
/// - Warm EQ with low body, rolled-off highs for analog warmth
/// - Deep vibrato at 5.0 Hz with wide spread for gospel shimmer
/// - Wide Gaussian detuning for thick, lush ensemble texture
/// - Formant preservation essential with 15-cent detuning
/// - Full humanization + prominent breath for emotional realism
/// - Bleed adds microphone crosstalk for cohesive ensemble
/// - Spatial depth: medium church with moderate distance
/// - Gentle dynamics preserve vocal expression and dynamics
fn build_gospel_choir(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Moderate alignment: gospel benefits from looser timing.
    config.alignment = Some(AlignmentConfig::new(0.5)?);

    // Warm gospel EQ: body at 200 Hz, gentle presence, rolled-off highs.
    config.eq = Some(EqConfig {
        low_freq: 200.0,
        low_gain_db: 2.0,
        mid_freq: 2800.0,
        mid_gain_db: 1.0,
        mid_q: 0.7,
        high_freq: 8000.0,
        high_gain_db: -2.0,
    });

    // Moderate de-essing.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.0,
        threshold_db: -20.0,
        max_reduction_db: -8.0,
        attack_sec: 0.001,
        release_sec: 0.060,
    });

    // Deep gospel vibrato: dramatic with wide per-voice spread.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.0,
        depth_cents: 45.0,
        rate_spread_hz: 0.6,
        depth_spread_cents: 15.0,
        onset_sec: 0.20,
    });

    // Wide Gaussian detuning for thick ensemble texture.
    config.detune = Some(DetuneConfig {
        cents_spread: 15.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Formant preservation: critical with 15-cent detuning.
    config.formant_preserve = Some(FormantPreserveConfig::default());

    // Full humanization for emotional gospel feel.
    config.humanize = Some(HumanizeConfig::default());

    // Prominent breath: louder, longer breaths fill the gospel space.
    config.breath = Some(
        BreathConfig::new()
            .with_noise_level(0.05)
            .with_duration_ms(150.0)
            .with_stagger_ms(45.0),
    );

    // Voice bleed: microphone crosstalk for cohesive gospel ensemble.
    config.bleed = Some(
        BleedConfig::new()
            .with_bleed_amount(0.06)
            .with_proximity_rolloff(1.5),
    );

    // Strong ensemble blending with harmonic alignment.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.55,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Wide stereo for gospel spaciousness.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.80));

    // Spatial depth: medium church space.
    config.spatial = Some(
        SpatialConfig::new()
            .with_room_size(12.0)
            .with_listener_distance(4.0),
    );

    // Gentle dynamics: preserve gospel expression and dynamics.
    config.dynamics = Some(DynamicsPreset::Gentle.to_config());

    config.limiter_enabled = true;
    Ok(config)
}

/// Ambient Pad: ethereal, wide stereo, heavy reverb, gentle freeze-like
/// sustain. 14 modules active.
///
/// Signal chain rationale:
/// - No alignment: ambient pads benefit from loose, drifting timing
/// - Dark EQ with heavy high roll-off for smooth, dreamy texture
/// - Slow vibrato at 3.5 Hz for gentle pitch drift
/// - Wide Gaussian detuning for thick unison pad
/// - Formant preservation prevents artifacts on wide detuning
/// - Envelope-only humanize for gentle swells
/// - Strong blending fuses voices into a single pad texture
/// - Maximum stereo width (1.8) with Haas delay for immersive spread
/// - Mastering dynamics: transparent, preserve pad dynamics
/// - Warm saturation for harmonic richness
/// - Large reverb + convolution for massive, hall-like space
/// - Spatial depth: very large room
fn build_ambient_pad(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Dark, smooth EQ: cut presence, roll off highs aggressively.
    config.eq = Some(EqConfig {
        low_freq: 250.0,
        low_gain_db: 1.5,
        mid_freq: 2000.0,
        mid_gain_db: -3.0,
        mid_q: 0.5,
        high_freq: 6000.0,
        high_gain_db: -4.0,
    });

    // Slow, dreamy vibrato: pitch drifts gently.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 3.5,
        depth_cents: 20.0,
        rate_spread_hz: 0.8,
        depth_spread_cents: 12.0,
        onset_sec: 0.50,
    });

    // Wide Gaussian detuning for thick unison pad.
    config.detune = Some(DetuneConfig {
        cents_spread: 18.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Formant preservation: essential with 18-cent detuning.
    config.formant_preserve = Some(FormantPreserveConfig::default());

    // Envelope-only humanization: gentle attack/release swells.
    config.humanize = Some(HumanizeConfig {
        enable_breath: false,
        enable_timing: false,
        enable_envelope: true,
        envelope: crate::kokoro_chorus_humanize::AmplitudeEnvelope {
            attack_sec: 0.080,
            hold_sec: 0.0,
            decay_sec: 0.060,
            sustain_level: 1.0,
            release_sec: 0.200,
        },
        ..HumanizeConfig::default()
    });

    // Strong blending: fuse voices into a single pad texture.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.70,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Maximum stereo width for immersive ambient spread.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(1.0));

    // Ultra-wide mid/side with Haas delay for immersive spaciousness.
    config.width = Some(
        StereoWidthConfig::new()
            .with_width(1.8)
            .with_bass_mono_freq(60.0)
            .with_haas_delay_ms(12.0),
    );

    // Spatial depth: very large ambient space.
    config.spatial = Some(
        SpatialConfig::new()
            .with_room_size(25.0)
            .with_listener_distance(6.0),
    );

    // Mastering dynamics: transparent, preserve pad dynamics.
    config.dynamics = Some(DynamicsPreset::Mastering.to_config());

    // Warm saturation: harmonic richness for pad texture.
    config.saturation = Some(
        SaturationConfig::new()
            .with_drive(0.12)
            .with_mix(0.35)
            .with_mode(SaturationMode::Warm)
            .with_output_gain_db(-0.5),
    );

    // Large reverb: maximum room and wet mix for ambient wash.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.40,
        room_size: 0.90,
        early_reflections: true,
        damping: 0.70,
    });

    // Convolution reverb: large hall IR for realistic ambient tail.
    config.convolution = Some(
        ConvolutionConfig::new()
            .with_wet_mix(0.30)
            .with_pre_delay_ms(25.0),
    );

    config.limiter_enabled = true;
    Ok(config)
}

/// Barbershop Quartet: tight harmony, minimal effects, close-mic character.
/// 9 modules active.
///
/// Signal chain rationale:
/// - Very tight alignment: barbershop requires precise unison/harmony
/// - Warm, natural EQ with gentle presence for close-mic clarity
/// - De-essing for close-mic sibilance control
/// - Very subtle detuning: just enough to avoid comb filtering
/// - Envelope-only humanize: no breath noise or timing drift
/// - Light blending: preserve individual voice character
/// - Narrow stereo: close, intimate positioning (quartet on stage)
/// - Mastering dynamics: transparent, preserve natural dynamics
/// - No reverb, no saturation, no spatial: dry, close-mic sound
fn build_barbershop_quartet(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Very tight alignment: barbershop demands lock-step timing.
    config.alignment = Some(AlignmentConfig::new(0.9)?);

    // Warm, natural EQ: proximity warmth, gentle presence, smooth highs.
    config.eq = Some(EqConfig {
        low_freq: 180.0,
        low_gain_db: 1.5,
        mid_freq: 2500.0,
        mid_gain_db: 1.0,
        mid_q: 0.8,
        high_freq: 8000.0,
        high_gain_db: -1.5,
    });

    // Gentle de-essing: close mic captures sibilance.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.0,
        threshold_db: -22.0,
        max_reduction_db: -8.0,
        attack_sec: 0.001,
        release_sec: 0.050,
    });

    // Minimal detuning: just enough to avoid comb filtering artifacts,
    // but not enough to smear the tight harmony.
    config.detune = Some(DetuneConfig {
        cents_spread: 3.0,
        distribution: DetuneDistribution::Uniform,
        seed: 0,
    });

    // Envelope-only humanization: clean, no breath or timing artifacts.
    config.humanize = Some(HumanizeConfig {
        enable_breath: false,
        enable_timing: false,
        enable_envelope: true,
        ..HumanizeConfig::default()
    });

    // Light blending: preserve individual voice character.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.25,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Narrow stereo: intimate quartet positioning, mono-compatible.
    config.stereo = Some(
        StereoChorusConfig::auto_layout(n_voices)?
            .with_stereo_width(0.40)
            .with_mono_compatible(true),
    );

    // Mastering dynamics: transparent, preserve natural quartet dynamics.
    config.dynamics = Some(DynamicsPreset::Mastering.to_config());

    config.limiter_enabled = true;
    Ok(config)
}

/// Opera Chorus: full dynamics, cathedral spatial, formant preservation.
/// 15 modules active.
///
/// Signal chain rationale:
/// - Moderate alignment: opera allows expressive timing variation
/// - Natural EQ with presence dip to prevent harshness in reverb tail
/// - Deep vibrato at classical rate (5.2 Hz) with wide spread
/// - Wide Gaussian detuning for rich choral texture
/// - Formant preservation critical for operatic vocal technique
/// - Full humanization + prominent breath for dramatic realism
/// - Voice bleed: stage crosstalk for cohesive ensemble
/// - Strong blending for unified choral mass
/// - Wide stereo for large-stage opera feel
/// - Large cathedral spatial with distant listener
/// - Convolution reverb for authentic hall ambience
/// - Gentle dynamics: preserve full operatic dynamic range
fn build_opera_chorus(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Moderate alignment: opera allows some timing freedom.
    config.alignment = Some(AlignmentConfig::new(0.55)?);

    // Natural EQ: slight low warmth, presence dip for reverb-friendly tone.
    config.eq = Some(EqConfig {
        low_freq: 200.0,
        low_gain_db: 0.5,
        mid_freq: 2500.0,
        mid_gain_db: -1.5,
        mid_q: 0.6,
        high_freq: 8500.0,
        high_gain_db: -1.0,
    });

    // Moderate de-essing: reverb amplifies sibilance.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 1.0,
        threshold_db: -20.0,
        max_reduction_db: -10.0,
        attack_sec: 0.001,
        release_sec: 0.060,
    });

    // Classical vibrato: 5.2 Hz at 40 cents with wide spread.
    config.vibrato = Some(VibratoConfig {
        rate_hz: 5.2,
        depth_cents: 40.0,
        rate_spread_hz: 0.5,
        depth_spread_cents: 12.0,
        onset_sec: 0.25,
    });

    // Wide Gaussian detuning for rich choral texture.
    config.detune = Some(DetuneConfig {
        cents_spread: 16.0,
        distribution: DetuneDistribution::Gaussian,
        seed: 0,
    });

    // Formant preservation: essential for operatic vocal technique.
    config.formant_preserve = Some(FormantPreserveConfig::default());

    // Full humanization for dramatic realism.
    config.humanize = Some(HumanizeConfig::default());

    // Prominent breath: dramatic breath sounds in cathedral space.
    config.breath = Some(
        BreathConfig::new()
            .with_noise_level(0.04)
            .with_duration_ms(130.0)
            .with_stagger_ms(40.0),
    );

    // Voice bleed: stage crosstalk for cohesive ensemble.
    config.bleed = Some(
        BleedConfig::new()
            .with_bleed_amount(0.05)
            .with_proximity_rolloff(1.8),
    );

    // Strong ensemble blending for unified choral mass.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.55,
        formant_preservation: true,
        harmonic_alignment: true,
        min_period: 30,
        max_period: 300,
    });

    // Wide stereo for large opera stage.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.90));

    // Cathedral spatial: large room, distant listener.
    config.spatial = Some(
        SpatialConfig::new()
            .with_room_size(20.0)
            .with_listener_distance(5.0),
    );

    // Convolution reverb: cathedral IR for authentic hall ambience.
    config.convolution = Some(
        ConvolutionConfig::new()
            .with_wet_mix(0.22)
            .with_pre_delay_ms(18.0),
    );

    // Gentle dynamics: preserve full operatic dynamic range.
    config.dynamics = Some(DynamicsPreset::Gentle.to_config());

    config.limiter_enabled = true;
    Ok(config)
}

/// Electronic Vocals: heavy pitch correction, exciter-like brightness,
/// saturation, and exaggerated stereo width. 13 modules active.
///
/// Signal chain rationale:
/// - Tight alignment for robotic, locked-in feel
/// - Bright EQ with boosted presence and air for cutting edge
/// - Aggressive de-essing before saturation to prevent harsh harmonics
/// - No vibrato: electronic vocals are straight-pitched
/// - Minimal uniform detuning for subtle chorus width
/// - Sharp transient boost for hard consonant attacks
/// - Light blending: preserve individual voice distinctness
/// - Wide stereo with ultra-wide mid/side and Haas delay
/// - Aggressive dynamics for heavy compression
/// - Tube saturation for edgy harmonic distortion
/// - Short bright reverb for electronic ambience
/// - Convolution for spatial gloss
fn build_electronic_vocals(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Tight alignment: electronic demands lock-step precision.
    config.alignment = Some(AlignmentConfig::new(0.85)?);

    // Bright, cutting EQ: boosted presence and air for electronic edge.
    config.eq = Some(EqConfig {
        low_freq: 120.0,
        low_gain_db: -2.0,
        mid_freq: 4000.0,
        mid_gain_db: 3.0,
        mid_q: 1.2,
        high_freq: 10000.0,
        high_gain_db: 1.5,
    });

    // Aggressive de-essing before saturation.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 5000.0,
        q: 1.5,
        threshold_db: -18.0,
        max_reduction_db: -14.0,
        attack_sec: 0.001,
        release_sec: 0.035,
    });

    // No vibrato: electronic vocals are straight-pitched.
    // config.vibrato = None; (already None by default)

    // Minimal uniform detuning for subtle chorus width.
    config.detune = Some(DetuneConfig {
        cents_spread: 6.0,
        distribution: DetuneDistribution::Uniform,
        seed: 0,
    });

    // Sharp transient boost for hard consonant attacks.
    config.transient = Some(
        TransientConfig::new()
            .with_attack(4.0)
            .with_sustain(-2.0)
            .with_sensitivity(2.5),
    );

    // Light blending: preserve individual voice distinctness.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.20,
        formant_preservation: false,
        harmonic_alignment: false,
        min_period: 30,
        max_period: 300,
    });

    // Wide stereo for spacious electronic sound.
    config.stereo = Some(StereoChorusConfig::auto_layout(n_voices)?.with_stereo_width(0.90));

    // Ultra-wide mid/side with Haas delay for electronic width.
    config.width = Some(
        StereoWidthConfig::new()
            .with_width(1.6)
            .with_bass_mono_freq(90.0)
            .with_haas_delay_ms(8.0),
    );

    // Aggressive dynamics: heavy compression for electronic loudness.
    config.dynamics = Some(DynamicsPreset::Aggressive.to_config());

    // Tube saturation: edgy harmonic distortion for electronic character.
    config.saturation = Some(
        SaturationConfig::new()
            .with_drive(0.30)
            .with_mix(0.55)
            .with_mode(SaturationMode::Tube)
            .with_output_gain_db(-1.0),
    );

    // Short bright reverb: electronic spatial ambience.
    config.reverb = Some(ReverbConfig {
        reverb_mix: 0.12,
        room_size: 0.20,
        early_reflections: true,
        damping: 0.35,
    });

    // Convolution reverb: electronic room gloss.
    config.convolution = Some(
        ConvolutionConfig::new()
            .with_wet_mix(0.10)
            .with_pre_delay_ms(5.0),
    );

    config.limiter_enabled = true;
    Ok(config)
}

/// Whisper: very quiet, close-mic, lots of breath, minimal dynamics.
/// 10 modules active.
///
/// Signal chain rationale:
/// - No alignment: whispers are inherently loose
/// - Very warm, dark EQ: proximity bass, heavy high roll-off
/// - Gentle de-essing: whispers have less sibilance energy
/// - No vibrato: whispers are straight-toned
/// - Minimal detuning: just enough to prevent phasing
/// - Full humanization with prominent breath noise
/// - Light blending for soft cohesion
/// - Very narrow stereo: close, intimate positioning
/// - Gentle width narrowing: bring everything to center
/// - Gentle dynamics: don't squash delicate whisper expression
fn build_whisper(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Very warm, dark EQ: proximity bass, rolled mids, heavy high cut.
    config.eq = Some(EqConfig {
        low_freq: 250.0,
        low_gain_db: 3.5,
        mid_freq: 2000.0,
        mid_gain_db: -2.5,
        mid_q: 0.6,
        high_freq: 5000.0,
        high_gain_db: -5.0,
    });

    // Gentle de-essing: whispers have less sibilance energy but close mic
    // still captures plosives and breath noise.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 6000.0,
        q: 0.8,
        threshold_db: -24.0,
        max_reduction_db: -6.0,
        attack_sec: 0.001,
        release_sec: 0.060,
    });

    // No vibrato: whispers are straight-toned.
    // config.vibrato = None; (already None by default)

    // Minimal detuning: just enough to prevent phasing artifacts.
    config.detune = Some(DetuneConfig {
        cents_spread: 2.0,
        distribution: DetuneDistribution::Uniform,
        seed: 0,
    });

    // Full humanization: whispers need breath and micro-timing.
    config.humanize = Some(HumanizeConfig {
        enable_breath: true,
        enable_timing: true,
        enable_envelope: true,
        envelope: crate::kokoro_chorus_humanize::AmplitudeEnvelope {
            attack_sec: 0.060,
            hold_sec: 0.0,
            decay_sec: 0.040,
            sustain_level: 1.0,
            release_sec: 0.180,
        },
        ..HumanizeConfig::default()
    });

    // Prominent breath: whispers are defined by breath noise.
    config.breath = Some(
        BreathConfig::new()
            .with_noise_level(0.06)
            .with_duration_ms(160.0)
            .with_stagger_ms(50.0),
    );

    // Light blending for soft cohesion.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.20,
        formant_preservation: true,
        harmonic_alignment: false,
        min_period: 30,
        max_period: 300,
    });

    // Very narrow stereo: close, intimate positioning.
    config.stereo = Some(
        StereoChorusConfig::auto_layout(n_voices)?
            .with_stereo_width(0.20)
            .with_mono_compatible(true),
    );

    // Gentle width narrowing: bring everything toward center.
    config.width = Some(
        StereoWidthConfig::new()
            .with_width(0.7)
            .with_bass_mono_freq(120.0),
    );

    // Gentle dynamics: preserve delicate whisper expression.
    config.dynamics = Some(DynamicsPreset::Gentle.to_config());

    config.limiter_enabled = true;
    Ok(config)
}

/// Announcer: broadcast-quality with heavy dynamics, de-essing, and
/// precise gain staging. 12 modules active.
///
/// Signal chain rationale:
/// - Very tight alignment: broadcast demands precise sync
/// - Broadcast EQ: low cut below 100 Hz, strong presence at 3.5 kHz
/// - Very aggressive de-essing: broadcast standards require sibilance control
/// - No vibrato: speech does not have natural vibrato
/// - Minimal detuning: just enough to prevent comb filtering
/// - Envelope-only humanization: clean attack/release
/// - Transient shaping: boost consonant clarity for intelligibility
/// - Light blending: preserve voice distinctness
/// - Moderate stereo: mono-compatible for radio playback
/// - Moderate width with bass mono for speaker compatibility
/// - Aggressive dynamics: tight compression for consistent loudness
/// - Console saturation: subtle broadcast warmth
fn build_announcer(n_voices: usize) -> Result<ChorusMasterConfig, KokoroError> {
    let mut config = ChorusMasterConfig::new(n_voices)?;

    // Very tight alignment: broadcast demands precision.
    config.alignment = Some(AlignmentConfig::new(0.85)?);

    // Broadcast EQ: low cut, strong presence, controlled highs.
    config.eq = Some(EqConfig {
        low_freq: 100.0,
        low_gain_db: -4.0,
        mid_freq: 3500.0,
        mid_gain_db: 3.0,
        mid_q: 1.0,
        high_freq: 10000.0,
        high_gain_db: -1.5,
    });

    // Very aggressive de-essing: broadcast sibilance standard.
    config.deesser = Some(DeEsserConfig {
        center_freq_hz: 5500.0,
        q: 1.5,
        threshold_db: -22.0,
        max_reduction_db: -16.0,
        attack_sec: 0.001,
        release_sec: 0.030,
    });

    // No vibrato: speech is straight-toned.
    // config.vibrato = None; (already None by default)

    // Minimal detuning: just enough to avoid comb filtering.
    config.detune = Some(DetuneConfig {
        cents_spread: 3.0,
        distribution: DetuneDistribution::Uniform,
        seed: 0,
    });

    // Envelope-only humanization: clean broadcast attack/release.
    config.humanize = Some(HumanizeConfig {
        enable_breath: false,
        enable_timing: false,
        enable_envelope: true,
        ..HumanizeConfig::default()
    });

    // Transient shaping: boost consonant clarity for intelligibility.
    config.transient = Some(
        TransientConfig::new()
            .with_attack(2.5)
            .with_sustain(-1.0)
            .with_sensitivity(1.8),
    );

    // Light blending: preserve individual voice character.
    config.blend = Some(EnsembleBlendConfig {
        blend_strength: 0.20,
        formant_preservation: true,
        harmonic_alignment: false,
        min_period: 30,
        max_period: 300,
    });

    // Moderate stereo with mono compatibility for radio.
    config.stereo = Some(
        StereoChorusConfig::auto_layout(n_voices)?
            .with_stereo_width(0.50)
            .with_mono_compatible(true),
    );

    // Moderate width with bass mono for speaker compatibility.
    config.width = Some(
        StereoWidthConfig::new()
            .with_width(1.1)
            .with_bass_mono_freq(100.0),
    );

    // Aggressive dynamics: tight compression for broadcast loudness.
    config.dynamics = Some(DynamicsPreset::Aggressive.to_config());

    // Console saturation: subtle broadcast warmth.
    config.saturation = Some(
        SaturationConfig::new()
            .with_drive(0.08)
            .with_mix(0.25)
            .with_mode(SaturationMode::Console)
            .with_output_gain_db(-0.3),
    );

    config.limiter_enabled = true;
    Ok(config)
}

// ---------------------------------------------------------------------------
// Chain validation
// ---------------------------------------------------------------------------

/// Severity level for chain validation warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningSeverity {
    /// Informational: not a problem, but worth knowing.
    Info,
    /// Warning: may cause audible artifacts in some conditions.
    Warning,
    /// Critical: very likely to cause audible problems.
    Critical,
}

/// A warning about a potentially problematic module combination.
#[derive(Debug, Clone)]
pub struct ChainWarning {
    /// Severity of this warning.
    pub severity: WarningSeverity,
    /// Human-readable description of the issue.
    pub message: String,
}

impl ChainWarning {
    fn new(severity: WarningSeverity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
        }
    }
}

/// Validate a [`ChorusMasterConfig`] for problematic module combinations.
///
/// Returns a list of warnings about combinations that may cause audible
/// artifacts. An empty list means no known problematic combinations.
///
/// Checks performed:
/// - Reverb + convolution = muddiness (reverb stacking)
/// - High stereo width (>1.5) + no mono compatibility = phase cancellation risk
/// - Saturation + high drive (>0.25) = harshness risk
/// - No limiter + high gain stages (saturation, dynamics, reverb) = clipping
/// - Very wide detuning (>20 cents) without formant preservation = chipmunk risk
/// - Spatial + stereo width = competing spatial processing
#[must_use]
pub fn validate_chain(config: &ChorusMasterConfig) -> Vec<ChainWarning> {
    let mut warnings = Vec::new();

    // Reverb + convolution stacking: both add reverb tails, risk of muddiness.
    if config.reverb.is_some() && config.convolution.is_some() {
        let reverb_wet = config.reverb.as_ref().map_or(0.0, |r| r.reverb_mix);
        let conv_wet = config.convolution.as_ref().map_or(0.0, |c| c.wet_mix);
        let combined_wet = reverb_wet + conv_wet;
        if combined_wet > 0.40 {
            warnings.push(ChainWarning::new(
                WarningSeverity::Warning,
                format!(
                    "Reverb + convolution combined wet mix ({:.0}%) is high; \
                     may cause muddiness. Consider reducing one.",
                    combined_wet * 100.0,
                ),
            ));
        } else {
            warnings.push(ChainWarning::new(
                WarningSeverity::Info,
                "Reverb + convolution are both active; monitor for muddiness.",
            ));
        }
    }

    // High width + not mono-compatible = phase cancellation risk.
    if let Some(ref width_cfg) = config.width {
        if width_cfg.width > 1.5 {
            let is_mono_safe = config.stereo.as_ref().map_or(false, |s| s.mono_compatible);
            if !is_mono_safe {
                warnings.push(ChainWarning::new(
                    WarningSeverity::Warning,
                    format!(
                        "Stereo width factor ({:.1}) is very high without mono \
                         compatibility enabled; risk of phase cancellation on \
                         mono playback systems.",
                        width_cfg.width,
                    ),
                ));
            }
        }
    }

    // Saturation + high drive = harshness risk.
    if let Some(ref sat) = config.saturation {
        if sat.drive > 0.25 {
            warnings.push(ChainWarning::new(
                WarningSeverity::Warning,
                format!(
                    "Saturation drive ({:.2}) is high; may cause harshness, \
                     especially combined with bright EQ or exciter.",
                    sat.drive,
                ),
            ));
        }
    }

    // No limiter + high gain stages = clipping risk.
    if !config.limiter_enabled {
        let gain_stages = [
            config.saturation.is_some(),
            config.dynamics.is_some(),
            config.reverb.is_some(),
            config.convolution.is_some(),
        ];
        let active_gain_stages: usize = gain_stages.iter().filter(|&&x| x).count();
        if active_gain_stages >= 2 {
            warnings.push(ChainWarning::new(
                WarningSeverity::Critical,
                format!(
                    "Limiter is disabled but {active_gain_stages} gain stages are active \
                     (saturation/dynamics/reverb/convolution); high risk \
                     of clipping.",
                ),
            ));
        }
    }

    // Wide detuning without formant preservation = chipmunk risk.
    if let Some(ref detune) = config.detune {
        if detune.cents_spread > 20.0 && config.formant_preserve.is_none() {
            warnings.push(ChainWarning::new(
                WarningSeverity::Warning,
                format!(
                    "Detuning spread ({:.0} cents) is wide without formant \
                     preservation; outer voices may sound unnatural.",
                    detune.cents_spread,
                ),
            ));
        }
    }

    // Spatial + stereo width = competing spatial processing.
    if config.spatial.is_some() && config.width.is_some() {
        if let Some(ref width_cfg) = config.width {
            if width_cfg.width > 1.3 {
                warnings.push(ChainWarning::new(
                    WarningSeverity::Info,
                    "Spatial processing and stereo width enhancement are both \
                     active; their spatial effects may interact unpredictably.",
                ));
            }
        }
    }

    warnings
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kokoro_chorus_pipeline::ChorusMasterPipeline;

    // -- Preset construction tests --

    #[test]
    fn test_all_presets_build_4_voices() {
        for &preset in VocalChainPreset::ALL {
            let config = preset
                .to_config(4)
                .unwrap_or_else(|e| panic!("preset {preset:?} should build with 4 voices: {e}"));
            config
                .validate()
                .unwrap_or_else(|e| panic!("preset {preset:?} config should validate: {e}"));
        }
    }

    #[test]
    fn test_all_presets_build_1_voice() {
        for &preset in VocalChainPreset::ALL {
            let config = preset
                .to_config(1)
                .unwrap_or_else(|e| panic!("preset {preset:?} with 1 voice: {e}"));
            config.validate().unwrap_or_else(|e| {
                panic!("preset {preset:?} with 1 voice should validate: {e}")
            });
        }
    }

    #[test]
    fn test_all_presets_build_32_voices() {
        for &preset in VocalChainPreset::ALL {
            let config = preset
                .to_config(32)
                .unwrap_or_else(|e| panic!("preset {preset:?} with 32 voices: {e}"));
            config.validate().unwrap_or_else(|e| {
                panic!("preset {preset:?} with 32 voices should validate: {e}")
            });
        }
    }

    #[test]
    fn test_zero_voices_fails() {
        for &preset in VocalChainPreset::ALL {
            assert!(
                preset.to_config(0).is_err(),
                "preset {preset:?} should fail with 0 voices",
            );
        }
    }

    #[test]
    fn test_33_voices_fails() {
        for &preset in VocalChainPreset::ALL {
            assert!(
                preset.to_config(33).is_err(),
                "preset {preset:?} should fail with 33 voices",
            );
        }
    }

    // -- Module count tests --

    #[test]
    fn test_each_preset_enables_at_least_3_modules() {
        for &preset in VocalChainPreset::ALL {
            let count = preset.module_count();
            assert!(
                count >= 3,
                "preset {preset:?} should enable at least 3 modules, got {count}",
            );
        }
    }

    #[test]
    fn test_pop_choir_module_count() {
        let count = VocalChainPreset::PopChoir.module_count();
        assert!(
            count >= 14,
            "PopChoir should enable many modules, got {count}",
        );
    }

    #[test]
    fn test_barbershop_quartet_minimal_modules() {
        let count = VocalChainPreset::BarbershopQuartet.module_count();
        assert!(
            count <= 12,
            "BarbershopQuartet should have relatively few modules, got {count}",
        );
    }

    // -- Naming tests --

    #[test]
    fn test_preset_names_are_unique() {
        let names: Vec<&str> = VocalChainPreset::ALL.iter().map(|p| p.name()).collect();
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
        for &preset in VocalChainPreset::ALL {
            assert!(
                !preset.description().is_empty(),
                "preset {preset:?} description must not be empty",
            );
        }
    }

    // -- Pipeline processing tests (4-voice audio, no NaN) --

    /// Generate a simple test signal: 0.5s of 440 Hz sine at 24 kHz.
    fn test_signal() -> Vec<f32> {
        let sr = crate::kokoro_tts::KOKORO_SAMPLE_RATE as f32;
        let n = (sr * 0.5) as usize; // 0.5 seconds
        (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_pop_choir_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::PopChoir.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        assert!(!right.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "PopChoir left[{i}] is non-finite: {l}");
            assert!(r.is_finite(), "PopChoir right[{i}] is non-finite: {r}");
        }
    }

    #[test]
    fn test_gospel_choir_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::GospelChoir.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "GospelChoir left[{i}] non-finite: {l}");
            assert!(r.is_finite(), "GospelChoir right[{i}] non-finite: {r}");
        }
    }

    #[test]
    fn test_ambient_pad_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::AmbientPad.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "AmbientPad left[{i}] non-finite: {l}");
            assert!(r.is_finite(), "AmbientPad right[{i}] non-finite: {r}");
        }
    }

    #[test]
    fn test_barbershop_quartet_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::BarbershopQuartet.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "BarbershopQuartet left[{i}] non-finite: {l}");
            assert!(
                r.is_finite(),
                "BarbershopQuartet right[{i}] non-finite: {r}"
            );
        }
    }

    #[test]
    fn test_opera_chorus_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::OperaChorus.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "OperaChorus left[{i}] non-finite: {l}");
            assert!(r.is_finite(), "OperaChorus right[{i}] non-finite: {r}");
        }
    }

    #[test]
    fn test_electronic_vocals_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::ElectronicVocals.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "ElectronicVocals left[{i}] non-finite: {l}");
            assert!(r.is_finite(), "ElectronicVocals right[{i}] non-finite: {r}");
        }
    }

    #[test]
    fn test_whisper_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::Whisper.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "Whisper left[{i}] non-finite: {l}");
            assert!(r.is_finite(), "Whisper right[{i}] non-finite: {r}");
        }
    }

    #[test]
    fn test_announcer_processes_without_nan() {
        let sig = test_signal();
        let voices = vec![sig.clone(), sig.clone(), sig.clone(), sig];
        let config = VocalChainPreset::Announcer.to_config(4).unwrap();
        let mut pipeline = ChorusMasterPipeline::new(config).unwrap();
        let (left, right) = pipeline.process(&voices).unwrap();
        assert!(!left.is_empty());
        for (i, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            assert!(l.is_finite(), "Announcer left[{i}] non-finite: {l}");
            assert!(r.is_finite(), "Announcer right[{i}] non-finite: {r}");
        }
    }

    // -- Chain validation tests --

    #[test]
    fn test_validate_catches_reverb_plus_convolution() {
        let config = VocalChainPreset::AmbientPad.to_config(4).unwrap();
        let warnings = validate_chain(&config);
        assert!(
            warnings
                .iter()
                .any(|w| w.message.contains("Reverb") || w.message.contains("reverb")),
            "AmbientPad should warn about reverb stacking: {warnings:?}",
        );
    }

    #[test]
    fn test_validate_catches_high_width_no_mono() {
        let config = VocalChainPreset::AmbientPad.to_config(4).unwrap();
        let warnings = validate_chain(&config);
        assert!(
            warnings
                .iter()
                .any(|w| w.message.contains("width") || w.message.contains("phase")),
            "AmbientPad should warn about high width: {warnings:?}",
        );
    }

    #[test]
    fn test_validate_catches_high_saturation_drive() {
        let config = VocalChainPreset::ElectronicVocals.to_config(4).unwrap();
        let warnings = validate_chain(&config);
        assert!(
            warnings
                .iter()
                .any(|w| w.message.contains("Saturation") || w.message.contains("saturation")),
            "ElectronicVocals should warn about high saturation drive: {warnings:?}",
        );
    }

    #[test]
    fn test_validate_catches_no_limiter_with_gain_stages() {
        // Build a config with saturation + dynamics + reverb but no limiter.
        let mut config = ChorusMasterConfig::new(4).unwrap();
        config.saturation = Some(
            SaturationConfig::new()
                .with_drive(0.2)
                .with_mix(0.5)
                .with_mode(SaturationMode::Tape)
                .with_output_gain_db(-0.5),
        );
        config.dynamics = Some(DynamicsPreset::Broadcast.to_config());
        config.reverb = Some(ReverbConfig {
            reverb_mix: 0.2,
            room_size: 0.5,
            early_reflections: true,
            damping: 0.5,
        });
        config.limiter_enabled = false;

        let warnings = validate_chain(&config);
        assert!(
            warnings
                .iter()
                .any(|w| w.severity == WarningSeverity::Critical && w.message.contains("clipping")),
            "Should produce critical clipping warning: {warnings:?}",
        );
    }

    #[test]
    fn test_validate_barbershop_has_minimal_warnings() {
        let config = VocalChainPreset::BarbershopQuartet.to_config(4).unwrap();
        let warnings = validate_chain(&config);
        // Barbershop is intentionally conservative; should have few/no warnings.
        let critical = warnings
            .iter()
            .filter(|w| w.severity == WarningSeverity::Critical)
            .count();
        assert_eq!(
            critical, 0,
            "BarbershopQuartet should have 0 critical warnings, got {critical}",
        );
    }

    #[test]
    fn test_validate_all_presets_no_critical() {
        // All built-in presets should be designed to avoid critical warnings.
        for &preset in VocalChainPreset::ALL {
            let config = preset.to_config(4).unwrap();
            let warnings = validate_chain(&config);
            let critical: Vec<_> = warnings
                .iter()
                .filter(|w| w.severity == WarningSeverity::Critical)
                .collect();
            assert!(
                critical.is_empty(),
                "preset {:?} should have no critical warnings, got: {:?}",
                preset,
                critical.iter().map(|w| &w.message).collect::<Vec<_>>(),
            );
        }
    }

    // -- VocalChain::from_preset equivalence test --

    #[test]
    fn test_from_preset_matches_to_config() {
        for &preset in VocalChainPreset::ALL {
            let via_chain = VocalChain::from_preset(preset, 4).unwrap();
            let via_method = preset.to_config(4).unwrap();
            // Both should produce configs with the same module count.
            assert_eq!(
                count_active_modules(&via_chain),
                count_active_modules(&via_method),
                "preset {preset:?}: from_preset and to_config should produce \
                 the same module count",
            );
        }
    }
}

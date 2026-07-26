// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Mix automation and scene manager for dynamic chorus transitions.
//!
//! Enables smooth, time-varying transitions between different chorus
//! configurations — for example, starting with an intimate 2-voice mix
//! and swelling to a full 8-voice cathedral chorus mid-song.
//!
//! The [`MixAutomator`] holds a "from" scene, a "to" scene, and an
//! interpolation cursor. Each call to [`MixAutomator::get_current_params`]
//! returns the interpolated [`MixParams`] at the requested sample offset.
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

/// Validate that `val` is finite and within `[lo, hi]`.
fn check_range(field: &'static str, val: f32, lo: f32, hi: f32) -> Result<(), KokoroError> {
    if !val.is_finite() || val < lo || val > hi {
        return Err(KokoroError::InvalidConfig {
            field,
            reason: format!("must be finite and in [{lo}, {hi}], got {val}"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Crossfade curve
// ---------------------------------------------------------------------------

/// Interpolation curve used when transitioning between scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum CrossfadeCurve {
    /// Linear interpolation (perceived loudness dip at midpoint).
    Linear,
    /// Hermite smoothstep (3t^2 - 2t^3). Perceptually even for gain ramps.
    #[default]
    SCurve,
    /// Constant-power cosine crossfade. Best for overlapping audio sources.
    EqualPower,
}


/// Apply the selected crossfade curve to a normalised `t` in [0, 1].
fn apply_curve(curve: CrossfadeCurve, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match curve {
        CrossfadeCurve::Linear => t,
        CrossfadeCurve::SCurve => t * t * (3.0 - 2.0 * t),
        CrossfadeCurve::EqualPower => (t * std::f32::consts::FRAC_PI_2).sin(),
    }
}

// ---------------------------------------------------------------------------
// Effect enables
// ---------------------------------------------------------------------------

/// Bitfield indicating which effects are active in a scene.
///
/// Effects snap on/off at the midpoint of the transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectEnables(pub u32);

impl EffectEnables {
    pub const EQ: u32 = 1 << 0;
    pub const DEESSER: u32 = 1 << 1;
    pub const VIBRATO: u32 = 1 << 2;
    pub const DETUNE: u32 = 1 << 3;
    pub const HUMANIZE: u32 = 1 << 4;
    pub const DYNAMICS: u32 = 1 << 5;
    pub const SATURATION: u32 = 1 << 6;
    pub const REVERB: u32 = 1 << 7;
    pub const SPATIAL: u32 = 1 << 8;
    /// All effects enabled.
    pub const ALL: Self = Self(0x1FF);
    /// No effects.
    pub const NONE: Self = Self(0);

    /// Check whether a specific effect bit is set.
    #[inline]
    pub fn is_enabled(self, bit: u32) -> bool {
        self.0 & bit != 0
    }

    /// Interpolate between two enable sets: snap at `t >= 0.5`.
    pub fn interpolate(from: Self, to: Self, t: f32) -> Self {
        if t < 0.5 {
            from
        } else {
            to
        }
    }
}

// ---------------------------------------------------------------------------
// Scene snapshot
// ---------------------------------------------------------------------------

/// Captures key mixer parameters at a point in time.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SceneSnapshot {
    /// Per-voice gain multipliers. Length must match voice count.
    pub per_voice_gains: Vec<f32>,
    /// Per-voice pan positions in [-1, 1]. Length must match voice count.
    pub per_voice_pans: Vec<f32>,
    /// Master output gain multiplier.
    pub master_gain: f32,
    /// Stereo width. 0 = mono, 1 = normal, 2 = hyper-wide.
    pub stereo_width: f32,
    /// Reverb wet/dry mix in [0, 1].
    pub reverb_mix: f32,
    /// Dynamics compressor threshold in dBFS.
    pub dynamics_threshold: f32,
    /// Which effects are active in this scene.
    pub effect_enables: EffectEnables,
}

impl SceneSnapshot {
    /// Create a new scene for `n_voices` with sensible defaults.
    pub fn new(n_voices: usize) -> Self {
        Self {
            per_voice_gains: vec![1.0; n_voices],
            per_voice_pans: default_pans(n_voices),
            master_gain: 1.0,
            stereo_width: 1.0,
            reverb_mix: 0.15,
            dynamics_threshold: -18.0,
            effect_enables: EffectEnables::ALL,
        }
    }

    /// Validate that all parameters are finite and in legal ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.per_voice_gains.is_empty() {
            return Err(KokoroError::InvalidConfig {
                field: "per_voice_gains",
                reason: "must have at least one voice".into(),
            });
        }
        if self.per_voice_gains.len() != self.per_voice_pans.len() {
            return Err(KokoroError::InvalidConfig {
                field: "per_voice_pans",
                reason: format!(
                    "gains len {} != pans len {}",
                    self.per_voice_gains.len(),
                    self.per_voice_pans.len(),
                ),
            });
        }
        for (i, &g) in self.per_voice_gains.iter().enumerate() {
            check_range("per_voice_gains", g, 0.0, 10.0).map_err(|_| {
                KokoroError::InvalidConfig {
                    field: "per_voice_gains",
                    reason: format!("voice {i}: gain {g} out of [0, 10]"),
                }
            })?;
        }
        for (i, &p) in self.per_voice_pans.iter().enumerate() {
            check_range("per_voice_pans", p, -1.0, 1.0).map_err(|_| {
                KokoroError::InvalidConfig {
                    field: "per_voice_pans",
                    reason: format!("voice {i}: pan {p} out of [-1, 1]"),
                }
            })?;
        }
        check_range("master_gain", self.master_gain, 0.0, 10.0)?;
        check_range("stereo_width", self.stereo_width, 0.0, 4.0)?;
        check_range("reverb_mix", self.reverb_mix, 0.0, 1.0)?;
        check_range("dynamics_threshold", self.dynamics_threshold, -96.0, 0.0)?;
        Ok(())
    }

    /// Number of voices in this snapshot.
    pub fn n_voices(&self) -> usize {
        self.per_voice_gains.len()
    }
}

/// Generate default pan positions spread evenly across the stereo field.
fn default_pans(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![0.0; n];
    }
    (0..n)
        .map(|i| (i as f32 / (n - 1) as f32) * 2.0 - 1.0)
        .collect()
}

// ---------------------------------------------------------------------------
// Interpolated mix parameters
// ---------------------------------------------------------------------------

/// Current interpolated parameters at a given sample position.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MixParams {
    /// Per-voice gain multipliers (interpolated).
    pub per_voice_gains: Vec<f32>,
    /// Per-voice pan positions (interpolated).
    pub per_voice_pans: Vec<f32>,
    /// Master gain multiplier (interpolated).
    pub master_gain: f32,
    /// Stereo width (interpolated).
    pub stereo_width: f32,
    /// Reverb wet/dry mix (interpolated).
    pub reverb_mix: f32,
    /// Dynamics threshold in dBFS (interpolated).
    pub dynamics_threshold: f32,
    /// Active effects (snapped at midpoint).
    pub effect_enables: EffectEnables,
    /// How far through the current transition (0 = start, 1 = end).
    pub transition_progress: f32,
}

// ---------------------------------------------------------------------------
// Automation timeline
// ---------------------------------------------------------------------------

/// A single keyframe in the automation timeline.
#[derive(Debug, Clone)]
pub struct TimelineKeyframe {
    /// Sample position at which this scene begins.
    pub sample_position: usize,
    /// Target scene parameters.
    pub scene: SceneSnapshot,
    /// Duration of transition INTO this scene, in milliseconds.
    pub transition_ms: f32,
}

/// Ordered list of keyframes defining scene changes over time.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct AutomationTimeline {
    /// Keyframes sorted by `sample_position`.
    pub keyframes: Vec<TimelineKeyframe>,
}

impl AutomationTimeline {
    /// Create an empty timeline.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe. Maintains sorted order by sample position.
    pub fn add_keyframe(
        &mut self,
        sample_position: usize,
        scene: SceneSnapshot,
        transition_ms: f32,
    ) {
        let kf = TimelineKeyframe {
            sample_position,
            scene,
            transition_ms,
        };
        let pos = self
            .keyframes
            .partition_point(|k| k.sample_position < sample_position);
        self.keyframes.insert(pos, kf);
    }

    /// Validate all keyframes.
    pub fn validate(&self) -> Result<(), KokoroError> {
        for (i, kf) in self.keyframes.iter().enumerate() {
            kf.scene
                .validate()
                .map_err(|e| KokoroError::InvalidConfig {
                    field: "timeline",
                    reason: format!("keyframe {i}: {e}"),
                })?;
            if !kf.transition_ms.is_finite() || kf.transition_ms < 0.0 {
                return Err(KokoroError::InvalidConfig {
                    field: "timeline",
                    reason: format!(
                        "keyframe {i}: transition_ms must be >= 0, got {}",
                        kf.transition_ms
                    ),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Automation config
// ---------------------------------------------------------------------------

/// Top-level configuration for mix automation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AutomationConfig {
    /// Named scene presets for quick recall.
    pub scenes: Vec<(String, SceneSnapshot)>,
    /// Default transition duration in milliseconds.
    pub default_transition_ms: f32,
    /// Crossfade interpolation curve.
    pub crossfade_curve: CrossfadeCurve,
}

impl Default for AutomationConfig {
    fn default() -> Self {
        Self {
            scenes: Vec::new(),
            default_transition_ms: 500.0,
            crossfade_curve: CrossfadeCurve::SCurve,
        }
    }
}

impl AutomationConfig {
    /// Create a config with a default transition time and S-curve.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set default transition time.
    #[must_use]
    pub fn with_transition_ms(mut self, ms: f32) -> Self {
        self.default_transition_ms = ms;
        self
    }

    /// Builder: set crossfade curve.
    #[must_use]
    pub fn with_curve(mut self, curve: CrossfadeCurve) -> Self {
        self.crossfade_curve = curve;
        self
    }

    /// Builder: register a named scene.
    #[must_use]
    pub fn with_scene(mut self, name: impl Into<String>, scene: SceneSnapshot) -> Self {
        self.scenes.push((name.into(), scene));
        self
    }

    /// Validate all registered scenes and parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        check_range(
            "default_transition_ms",
            self.default_transition_ms,
            0.0,
            f32::MAX,
        )?;
        for (name, scene) in &self.scenes {
            scene.validate().map_err(|e| KokoroError::InvalidConfig {
                field: "scenes",
                reason: format!("scene '{name}': {e}"),
            })?;
        }
        Ok(())
    }

    /// Look up a named scene.
    pub fn find_scene(&self, name: &str) -> Option<&SceneSnapshot> {
        self.scenes.iter().find(|(n, _)| n == name).map(|(_, s)| s)
    }
}

// ---------------------------------------------------------------------------
// Mix automator
// ---------------------------------------------------------------------------

/// Interpolates between scene snapshots over time.
///
/// When [`set_scene`](MixAutomator::set_scene) is called, the automator begins
/// a smooth transition from the current parameters to the new scene.
pub struct MixAutomator {
    from: SceneSnapshot,
    to: SceneSnapshot,
    curve: CrossfadeCurve,
    transition_start: usize,
    transition_samples: usize,
    sample_clock: usize,
    sample_rate: usize,
}

impl MixAutomator {
    /// Create a new automator at the given initial scene.
    pub fn new(initial: SceneSnapshot, curve: CrossfadeCurve, sample_rate: usize) -> Self {
        Self {
            from: initial.clone(),
            to: initial,
            curve,
            transition_start: 0,
            transition_samples: 0,
            sample_clock: 0,
            sample_rate,
        }
    }

    /// Create a new automator at Kokoro sample rate (24 kHz).
    pub fn new_kokoro(initial: SceneSnapshot, curve: CrossfadeCurve) -> Self {
        Self::new(initial, curve, KOKORO_SAMPLE_RATE)
    }

    /// Begin a smooth transition to a new scene.
    ///
    /// If a transition is already in progress, the current interpolated
    /// state becomes the new "from" scene (seamless retrigger).
    pub fn set_scene(&mut self, scene: SceneSnapshot, transition_ms: f32) {
        let current = self.get_current_params(0);
        self.from = SceneSnapshot {
            per_voice_gains: current.per_voice_gains,
            per_voice_pans: current.per_voice_pans,
            master_gain: current.master_gain,
            stereo_width: current.stereo_width,
            reverb_mix: current.reverb_mix,
            dynamics_threshold: current.dynamics_threshold,
            effect_enables: current.effect_enables,
        };
        self.to = scene;
        self.transition_start = self.sample_clock;
        self.transition_samples = ms_to_samples(transition_ms, self.sample_rate);
    }

    /// Advance the sample clock by `n` samples.
    pub fn advance(&mut self, n: usize) {
        self.sample_clock += n;
    }

    /// Reset the automator to a specific scene with no transition.
    pub fn reset(&mut self, scene: SceneSnapshot) {
        self.from = scene.clone();
        self.to = scene;
        self.transition_start = self.sample_clock;
        self.transition_samples = 0;
    }

    /// Whether a transition is currently in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition_samples > 0
            && self.sample_clock.saturating_sub(self.transition_start) < self.transition_samples
    }

    /// Get the interpolated mix parameters at the given sample offset
    /// from the current clock position.
    pub fn get_current_params(&self, sample_offset: usize) -> MixParams {
        let pos = self.sample_clock + sample_offset;
        let t_raw = if self.transition_samples == 0 {
            1.0
        } else {
            let elapsed = pos.saturating_sub(self.transition_start);
            (elapsed as f32 / self.transition_samples as f32).clamp(0.0, 1.0)
        };
        let t = apply_curve(self.curve, t_raw);

        let max_voices = self.from.n_voices().max(self.to.n_voices());
        let mut gains = Vec::with_capacity(max_voices);
        let mut pans = Vec::with_capacity(max_voices);
        for i in 0..max_voices {
            let gf = self.from.per_voice_gains.get(i).copied().unwrap_or(0.0);
            let gt = self.to.per_voice_gains.get(i).copied().unwrap_or(0.0);
            gains.push(lerp(gf, gt, t));
            let pf = self.from.per_voice_pans.get(i).copied().unwrap_or(0.0);
            let pt = self.to.per_voice_pans.get(i).copied().unwrap_or(0.0);
            pans.push(lerp(pf, pt, t));
        }

        MixParams {
            per_voice_gains: gains,
            per_voice_pans: pans,
            master_gain: lerp(self.from.master_gain, self.to.master_gain, t),
            stereo_width: lerp(self.from.stereo_width, self.to.stereo_width, t),
            reverb_mix: lerp(self.from.reverb_mix, self.to.reverb_mix, t),
            dynamics_threshold: lerp(self.from.dynamics_threshold, self.to.dynamics_threshold, t),
            effect_enables: EffectEnables::interpolate(
                self.from.effect_enables,
                self.to.effect_enables,
                t,
            ),
            transition_progress: t_raw,
        }
    }

    /// Apply per-voice gain automation to a slice of voice buffers.
    ///
    /// `voices[v]` is the audio buffer for voice `v`. Samples in
    /// `[offset .. offset+length]` are scaled by the interpolated gain.
    pub fn process_gains(&self, voices: &mut [Vec<f32>], offset: usize, length: usize) {
        for (v, buf) in voices.iter_mut().enumerate() {
            let end = (offset + length).min(buf.len());
            for i in offset..end {
                let params = self.get_current_params(i.saturating_sub(offset));
                let gain = params.per_voice_gains.get(v).copied().unwrap_or(0.0);
                buf[i] *= gain;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Presets (extracted to stay under 500-line limit)
// ---------------------------------------------------------------------------

#[path = "kokoro_chorus_automation_presets.rs"]
mod presets;
pub use presets::{build_to_chorus, dynamic_swell, fade_to_intimate};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linear interpolation between `a` and `b` at fraction `t`.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Convert milliseconds to samples at the given sample rate.
#[inline]
fn ms_to_samples(ms: f32, sample_rate: usize) -> usize {
    ((ms / 1000.0) * sample_rate as f32).round() as usize
}

#[cfg(test)]
#[path = "kokoro_chorus_automation_tests.rs"]
mod tests;

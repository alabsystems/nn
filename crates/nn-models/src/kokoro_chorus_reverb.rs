// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Spatial reverb and room modeling for chorus audio mixing.
//!
//! Implements a Schroeder reverb (4 comb filters + 2 allpass filters) with
//! per-voice early reflections for spatial width. Operates on CPU-side PCM
//! after the chorus voices are mixed to stereo.
//!
//! # Architecture
//!
//! ```text
//! Mixed stereo audio
//!   → Early reflections (per-voice Haas-effect delays based on pan position)
//!   → Late reverb (Schroeder: 4 comb → 2 allpass, per-channel)
//!   → Dry/wet mix (reverb_mix controls blend ratio)
//!   → Output stereo audio
//! ```
//!
//! # References
//!
//! - Schroeder, M.R. (1962). "Natural Sounding Artificial Reverberation."
//!   Journal of the Audio Engineering Society, 10(3), 219-223.
//! - Haas, H. (1951). "The influence of a single echo on the audibility
//!   of speech." Acustica, 1, 49-58.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Reverb configuration
// ---------------------------------------------------------------------------

/// Configuration for spatial reverb applied to chorus output.
///
/// Controls the dry/wet ratio, room size (late reverb decay), early
/// reflections, and high-frequency damping. Built via method chaining.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ReverbConfig {
    /// Dry/wet ratio: 0.0 = fully dry, 1.0 = fully wet.
    ///
    /// Default: `0.15`. For chorus, subtle reverb (0.10-0.25) adds spatial
    /// depth without muddying the voices.
    pub reverb_mix: f32,

    /// Room size controlling late reverb decay length (0.0-1.0).
    ///
    /// Default: `0.3`. Maps to comb filter feedback gain:
    /// `feedback = 0.7 + 0.28 * room_size`. Larger values = longer tail.
    /// 0.0 = small room (~0.2s RT60), 1.0 = large hall (~2s RT60).
    pub room_size: f32,

    /// Enable Haas-effect early reflections based on voice pan positions.
    ///
    /// Default: `true`. Adds 1-15ms delayed copies per voice, scaled by
    /// pan position, creating a wider spatial image. Disabled for mono
    /// output or when spatial width is not desired.
    pub early_reflections: bool,

    /// High-frequency damping factor (0.0-1.0).
    ///
    /// Default: `0.5`. Controls how quickly high frequencies decay in the
    /// reverb tail. Higher values = more high-frequency absorption (warmer
    /// sound). Applied as a one-pole lowpass in each comb filter.
    pub damping: f32,
}

impl Default for ReverbConfig {
    fn default() -> Self {
        Self {
            reverb_mix: 0.15,
            room_size: 0.3,
            early_reflections: true,
            damping: 0.5,
        }
    }
}

impl ReverbConfig {
    /// Create a new reverb config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the dry/wet mix ratio.
    #[must_use]
    pub fn with_reverb_mix(mut self, mix: f32) -> Self {
        self.reverb_mix = mix;
        self
    }

    /// Set the room size (0.0 = small room, 1.0 = large hall).
    #[must_use]
    pub fn with_room_size(mut self, size: f32) -> Self {
        self.room_size = size;
        self
    }

    /// Enable or disable early reflections.
    #[must_use]
    pub fn with_early_reflections(mut self, enable: bool) -> Self {
        self.early_reflections = enable;
        self
    }

    /// Set high-frequency damping (0.0 = bright, 1.0 = dark/warm).
    #[must_use]
    pub fn with_damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Validate that all parameters are within valid ranges.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.reverb_mix.is_finite() || !(0.0..=1.0).contains(&self.reverb_mix) {
            return Err(KokoroError::InvalidConfig {
                field: "reverb_mix",
                reason: format!(
                    "reverb_mix = {}: must be finite and in [0.0, 1.0]",
                    self.reverb_mix,
                ),
            });
        }
        if !self.room_size.is_finite() || !(0.0..=1.0).contains(&self.room_size) {
            return Err(KokoroError::InvalidConfig {
                field: "room_size",
                reason: format!(
                    "room_size = {}: must be finite and in [0.0, 1.0]",
                    self.room_size,
                ),
            });
        }
        if !self.damping.is_finite() || !(0.0..=1.0).contains(&self.damping) {
            return Err(KokoroError::InvalidConfig {
                field: "damping",
                reason: format!(
                    "damping = {}: must be finite and in [0.0, 1.0]",
                    self.damping,
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Schroeder reverb: comb filter delays (in samples at 24kHz)
// ---------------------------------------------------------------------------

/// Comb filter delay lengths in samples at 24kHz.
///
/// These are mutually prime to avoid metallic resonance. Based on Schroeder's
/// original design scaled to 24kHz sample rate.
const COMB_DELAYS: [usize; 4] = [1116, 1188, 1277, 1356];

/// Allpass filter delay lengths in samples at 24kHz.
const ALLPASS_DELAYS: [usize; 2] = [225, 131];

/// Allpass feedback coefficient (standard Schroeder value).
const ALLPASS_FEEDBACK: f32 = 0.5;

// ---------------------------------------------------------------------------
// Comb filter with damping
// ---------------------------------------------------------------------------

/// Lowpass-feedback comb filter for Schroeder reverb.
///
/// Each comb filter is a delay line with feedback and a one-pole lowpass
/// in the feedback path for high-frequency damping.
struct CombFilter {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
    damping: f32,
    damp_state: f32,
}

impl CombFilter {
    fn new(delay_len: usize, feedback: f32, damping: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_len],
            index: 0,
            feedback,
            damping,
            damp_state: 0.0,
        }
    }

    /// Process one sample through the comb filter.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let output = self.buffer[self.index];

        // One-pole lowpass in feedback path: y[n] = (1-d)*x[n] + d*y[n-1]
        self.damp_state = output * (1.0 - self.damping) + self.damp_state * self.damping;

        self.buffer[self.index] = input + self.damp_state * self.feedback;
        self.index = (self.index + 1) % self.buffer.len();

        output
    }
}

// ---------------------------------------------------------------------------
// Allpass filter
// ---------------------------------------------------------------------------

/// Allpass filter for Schroeder reverb.
///
/// Diffuses the reverb tail without changing the frequency balance.
struct AllpassFilter {
    buffer: Vec<f32>,
    index: usize,
    feedback: f32,
}

impl AllpassFilter {
    fn new(delay_len: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_len],
            index: 0,
            feedback,
        }
    }

    /// Process one sample through the allpass filter.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        let buffered = self.buffer[self.index];
        let output = -input + buffered;
        self.buffer[self.index] = input + buffered * self.feedback;
        self.index = (self.index + 1) % self.buffer.len();
        output
    }
}

// ---------------------------------------------------------------------------
// Schroeder reverberator (single channel)
// ---------------------------------------------------------------------------

/// Single-channel Schroeder reverberator: 4 comb filters in parallel
/// followed by 2 allpass filters in series.
struct SchroederChannel {
    combs: [CombFilter; 4],
    allpasses: [AllpassFilter; 2],
}

impl SchroederChannel {
    fn new(feedback: f32, damping: f32) -> Self {
        Self {
            combs: [
                CombFilter::new(COMB_DELAYS[0], feedback, damping),
                CombFilter::new(COMB_DELAYS[1], feedback, damping),
                CombFilter::new(COMB_DELAYS[2], feedback, damping),
                CombFilter::new(COMB_DELAYS[3], feedback, damping),
            ],
            allpasses: [
                AllpassFilter::new(ALLPASS_DELAYS[0], ALLPASS_FEEDBACK),
                AllpassFilter::new(ALLPASS_DELAYS[1], ALLPASS_FEEDBACK),
            ],
        }
    }

    /// Process one sample.
    #[inline]
    fn process(&mut self, input: f32) -> f32 {
        // Sum parallel comb filters.
        let mut output = 0.0f32;
        for comb in &mut self.combs {
            output += comb.process(input);
        }
        // Series allpass filters.
        for allpass in &mut self.allpasses {
            output = allpass.process(output);
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Stereo Schroeder reverb
// ---------------------------------------------------------------------------

/// Stereo Schroeder reverb processor.
///
/// Uses slightly different feedback for left/right channels to create
/// natural decorrelation between channels.
pub struct StereoReverb {
    left: SchroederChannel,
    right: SchroederChannel,
    mix: f32,
}

impl StereoReverb {
    /// Create a new stereo reverb from a `ReverbConfig`.
    pub fn new(config: &ReverbConfig) -> Self {
        let feedback = 0.7 + 0.28 * config.room_size;
        // Slightly different feedback for left/right to decorrelate.
        let feedback_l = feedback;
        let feedback_r = feedback * 0.98;
        Self {
            left: SchroederChannel::new(feedback_l, config.damping),
            right: SchroederChannel::new(feedback_r, config.damping),
            mix: config.reverb_mix,
        }
    }

    /// Process interleaved stereo audio in-place.
    ///
    /// Input/output format: `[L0, R0, L1, R1, ...]`.
    pub fn process_stereo(&mut self, buffer: &mut [f32]) {
        let num_frames = buffer.len() / 2;
        for i in 0..num_frames {
            let dry_l = buffer[i * 2];
            let dry_r = buffer[i * 2 + 1];
            let wet_l = self.left.process(dry_l);
            let wet_r = self.right.process(dry_r);
            buffer[i * 2] = dry_l * (1.0 - self.mix) + wet_l * self.mix;
            buffer[i * 2 + 1] = dry_r * (1.0 - self.mix) + wet_r * self.mix;
        }
    }

    /// Process mono audio in-place.
    pub fn process_mono(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            let dry = *sample;
            let wet = self.left.process(dry);
            *sample = dry * (1.0 - self.mix) + wet * self.mix;
        }
    }
}

// ---------------------------------------------------------------------------
// Early reflections
// ---------------------------------------------------------------------------

/// Maximum early reflection delay in seconds (15ms).
const MAX_EARLY_DELAY_SEC: f32 = 0.015;

/// Minimum early reflection delay in seconds (1ms).
const MIN_EARLY_DELAY_SEC: f32 = 0.001;

/// Number of early reflections per voice.
const NUM_EARLY_REFLECTIONS: usize = 3;

/// Early reflection gain attenuation per tap (each successive tap is quieter).
const EARLY_REFLECTION_DECAY: f32 = 0.7;

/// Compute early reflection delays for a voice based on its pan position.
///
/// Voices panned further from center get longer delays (simulating distance
/// from the nearest wall). Returns delay/gain pairs for left and right channels.
///
/// The Haas effect (1-30ms delay) creates a perception of spaciousness
/// without being perceived as a distinct echo.
fn compute_early_reflection_taps(pan: f32, sample_rate: f32) -> Vec<(usize, f32, usize, f32)> {
    // Pan position determines asymmetry: a voice panned left gets shorter
    // left-channel delays (closer wall) and longer right-channel delays.
    let pan_abs = pan.abs().clamp(0.0, 1.0);
    let base_delay_sec =
        MIN_EARLY_DELAY_SEC + pan_abs * (MAX_EARLY_DELAY_SEC - MIN_EARLY_DELAY_SEC);

    let mut taps = Vec::with_capacity(NUM_EARLY_REFLECTIONS);
    for tap_idx in 0..NUM_EARLY_REFLECTIONS {
        let tap_scale = 1.0 + tap_idx as f32 * 0.7;
        let gain = EARLY_REFLECTION_DECAY.powi(tap_idx as i32 + 1);

        // Left channel delay: shorter when voice is panned left.
        let left_delay_sec = base_delay_sec * tap_scale * (1.0 - pan * 0.3).clamp(0.5, 1.5);
        let left_delay_samples = (left_delay_sec * sample_rate).round() as usize;

        // Right channel delay: shorter when voice is panned right.
        let right_delay_sec = base_delay_sec * tap_scale * (1.0 + pan * 0.3).clamp(0.5, 1.5);
        let right_delay_samples = (right_delay_sec * sample_rate).round() as usize;

        taps.push((left_delay_samples, gain, right_delay_samples, gain));
    }
    taps
}

/// Apply per-voice early reflections to interleaved stereo audio.
///
/// For each voice, adds delayed copies of the voice's contribution to the
/// stereo buffer based on the voice's pan position. This creates a Haas-effect
/// spatial widening that makes voices sound like they are in a real space.
///
/// # Arguments
///
/// * `stereo_buffer` - Interleaved stereo output buffer `[L0, R0, L1, R1, ...]`.
/// * `voice_audio` - Per-voice mono PCM buffers.
/// * `pans` - Per-voice pan positions in [-1.0, 1.0].
/// * `gains` - Per-voice gain multipliers.
pub(crate) fn apply_early_reflections(
    stereo_buffer: &mut [f32],
    voice_audio: &[&[f32]],
    pans: &[f32],
    gains: &[f32],
) {
    let sample_rate = KOKORO_SAMPLE_RATE as f32;
    let num_frames = stereo_buffer.len() / 2;

    for (voice_idx, pcm) in voice_audio.iter().enumerate() {
        let pan = pans.get(voice_idx).copied().unwrap_or(0.0);
        let gain = gains.get(voice_idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        if gain < 1e-6 {
            continue;
        }

        let taps = compute_early_reflection_taps(pan, sample_rate);
        // Reflection gain is relative to the voice gain (attenuated).
        let reflection_gain = gain * 0.3;

        for (left_delay, left_gain, right_delay, right_gain) in &taps {
            for (i, &sample) in pcm.iter().enumerate() {
                // Left channel reflection.
                let left_dst = i + left_delay;
                if left_dst < num_frames {
                    stereo_buffer[left_dst * 2] += sample * reflection_gain * left_gain;
                }
                // Right channel reflection.
                let right_dst = i + right_delay;
                if right_dst < num_frames {
                    stereo_buffer[right_dst * 2 + 1] += sample * reflection_gain * right_gain;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API: apply reverb to mixed chorus output
// ---------------------------------------------------------------------------

/// Apply spatial reverb to a mixed chorus stereo buffer.
///
/// Applies early reflections (if configured and voice data is available),
/// then late Schroeder reverb. The dry/wet ratio controls the blend.
///
/// # Arguments
///
/// * `buffer` - Audio output (modified in-place). Stereo: `[L0, R0, ...]`.
/// * `config` - Reverb configuration.
/// * `is_stereo` - Whether the buffer is interleaved stereo.
/// * `voice_audio` - Optional per-voice mono PCM (needed for early reflections).
/// * `pans` - Optional per-voice pan positions (needed for early reflections).
/// * `gains` - Optional per-voice gains (needed for early reflections).
pub(crate) fn apply_reverb(
    buffer: &mut [f32],
    config: &ReverbConfig,
    is_stereo: bool,
    voice_audio: Option<&[&[f32]]>,
    pans: Option<&[f32]>,
    gains: Option<&[f32]>,
) -> Result<(), KokoroError> {
    config.validate()?;

    // Skip if fully dry.
    if config.reverb_mix < 1e-6 {
        return Ok(());
    }

    // Apply early reflections if stereo and we have per-voice data.
    if is_stereo && config.early_reflections {
        if let (Some(voices), Some(pan_positions), Some(voice_gains)) = (voice_audio, pans, gains) {
            apply_early_reflections(buffer, voices, pan_positions, voice_gains);
        }
    }

    // Apply late Schroeder reverb.
    let mut reverb = StereoReverb::new(config);
    if is_stereo {
        reverb.process_stereo(buffer);
    } else {
        reverb.process_mono(buffer);
    }

    Ok(())
}

#[cfg(test)]
#[path = "kokoro_chorus_reverb_tests.rs"]
mod tests;

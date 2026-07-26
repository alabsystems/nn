// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Stereo imaging with constant-power panning for Kokoro chorus.
//!
//! Provides per-voice positioning in the stereo field using sin/cos
//! constant-power pan law. Each voice gets a distinct position (LCR or
//! custom), creating the perception of multiple singers on a stage.
//!
//! # Constant-power pan law
//!
//! Linear panning creates a 3dB perceived loudness dip at center because
//! `0.5 + 0.5 = 1.0` in amplitude but power is `0.5^2 + 0.5^2 = 0.5`.
//! The sin/cos law maintains constant power across all pan positions:
//!
//! ```text
//! angle = (pan + 1) * 0.5 * pi/2    // maps [-1, 1] -> [0, pi/2]
//! left_gain  = cos(angle)
//! right_gain = sin(angle)
//! // cos^2 + sin^2 = 1 for all angles (Pythagorean identity)
//! ```
//!
//! # Mono compatibility
//!
//! When `mono_compatible` is set, the stereo mix is verified to produce
//! an L+R sum close to the equivalent mono mix. The constant-power law
//! ensures `L + R = cos(a) + sin(a)` which ranges from 1.0 (at extremes)
//! to sqrt(2) ~ 1.414 (at center). A normalization factor of
//! `1 / sqrt(2)` can be applied per voice to keep L+R ~ 1.0 everywhere.
//!
//! Part of #4264, #3351.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Stereo position
// ---------------------------------------------------------------------------

/// Named stereo positions for voice placement in the stereo field.
///
/// Maps to a pan value in [-1.0, 1.0] for the constant-power panner.
/// Use `Custom(f32)` for arbitrary positions not covered by the named
/// presets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StereoPosition {
    /// Hard left (-1.0).
    Left,
    /// Between center and left (-0.5).
    CenterLeft,
    /// Dead center (0.0).
    Center,
    /// Between center and right (0.5).
    CenterRight,
    /// Hard right (1.0).
    Right,
    /// Custom position in [-1.0, 1.0].
    Custom(f32),
}

impl StereoPosition {
    /// Convert this position to a pan value in [-1.0, 1.0].
    #[must_use]
    pub fn to_pan(self) -> f32 {
        match self {
            Self::Left => -1.0,
            Self::CenterLeft => -0.5,
            Self::Center => 0.0,
            Self::CenterRight => 0.5,
            Self::Right => 1.0,
            Self::Custom(pan) => pan.clamp(-1.0, 1.0),
        }
    }

    /// Create a position from a pan value in [-1.0, 1.0].
    ///
    /// Returns the nearest named position if the value is within epsilon
    /// of a named position, otherwise returns `Custom(pan)`.
    #[must_use]
    pub fn from_pan(pan: f32) -> Self {
        const EPS: f32 = 0.01;
        let clamped = pan.clamp(-1.0, 1.0);
        if (clamped - (-1.0)).abs() < EPS {
            Self::Left
        } else if (clamped - (-0.5)).abs() < EPS {
            Self::CenterLeft
        } else if clamped.abs() < EPS {
            Self::Center
        } else if (clamped - 0.5).abs() < EPS {
            Self::CenterRight
        } else if (clamped - 1.0).abs() < EPS {
            Self::Right
        } else {
            Self::Custom(clamped)
        }
    }
}

// ---------------------------------------------------------------------------
// Default voice layouts
// ---------------------------------------------------------------------------

/// Generate a default stereo layout for `n` voices spread across the field.
///
/// Produces a natural stage layout:
/// - 1 voice: Center
/// - 2 voices: CenterLeft, CenterRight
/// - 3 voices: Left, Center, Right (classic LCR)
/// - 4 voices: Left, CenterLeft, CenterRight, Right
/// - 5 voices: Left, CenterLeft, Center, CenterRight, Right
/// - 6+ voices: evenly spaced from -1.0 to 1.0
#[must_use]
pub fn default_voice_layout(n_voices: usize) -> Vec<StereoPosition> {
    match n_voices {
        0 => Vec::new(),
        1 => vec![StereoPosition::Center],
        2 => vec![StereoPosition::CenterLeft, StereoPosition::CenterRight],
        3 => vec![
            StereoPosition::Left,
            StereoPosition::Center,
            StereoPosition::Right,
        ],
        4 => vec![
            StereoPosition::Left,
            StereoPosition::CenterLeft,
            StereoPosition::CenterRight,
            StereoPosition::Right,
        ],
        5 => vec![
            StereoPosition::Left,
            StereoPosition::CenterLeft,
            StereoPosition::Center,
            StereoPosition::CenterRight,
            StereoPosition::Right,
        ],
        n => {
            // Evenly space across [-1.0, 1.0].
            (0..n)
                .map(|i| {
                    let pan = if n == 1 {
                        0.0
                    } else {
                        -1.0 + 2.0 * (i as f32) / (n - 1) as f32
                    };
                    StereoPosition::from_pan(pan)
                })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Stereo panner
// ---------------------------------------------------------------------------

/// Constant-power stereo panner using sin/cos pan law.
///
/// Given a mono signal and a pan position, computes left and right gains
/// such that total power is preserved across all pan positions.
///
/// The panner supports a stereo width multiplier that scales pan positions
/// toward center (width=0.0 is mono, width=1.0 is full stereo).
#[derive(Debug, Clone)]
pub struct StereoPanner {
    /// Stereo width multiplier in [0.0, 1.0].
    width: f32,
}

impl Default for StereoPanner {
    fn default() -> Self {
        Self { width: 1.0 }
    }
}

impl StereoPanner {
    /// Create a new panner with the given stereo width.
    ///
    /// Width is clamped to [0.0, 1.0]. At 0.0 all voices collapse to
    /// center (mono). At 1.0 voices use their full pan positions.
    #[must_use]
    pub fn new(width: f32) -> Self {
        Self {
            width: width.clamp(0.0, 1.0),
        }
    }

    /// Compute left and right gains for a given pan position.
    ///
    /// Returns `(left_gain, right_gain)` where `left^2 + right^2 = 1.0`
    /// (constant power). The pan is first scaled by the stereo width.
    ///
    /// Pan range: -1.0 (hard left) to 1.0 (hard right).
    #[must_use]
    pub fn pan_gains(&self, pan: f32) -> (f32, f32) {
        let effective_pan = (pan * self.width).clamp(-1.0, 1.0);
        constant_power_pan(effective_pan)
    }

    /// Get the current stereo width.
    #[must_use]
    pub fn width(&self) -> f32 {
        self.width
    }
}

/// Compute constant-power pan gains for a pan position in [-1.0, 1.0].
///
/// Uses sin/cos pan law:
/// - `angle = (pan + 1) * 0.5 * pi/2`  maps [-1, 1] to [0, pi/2]
/// - `left_gain  = cos(angle)`
/// - `right_gain = sin(angle)`
///
/// At center (pan=0): both gains = cos(pi/4) = sin(pi/4) ~ 0.707
/// At hard left (pan=-1): left=1.0, right=0.0
/// At hard right (pan=1): left=0.0, right=1.0
#[inline]
#[must_use]
pub fn constant_power_pan(pan: f32) -> (f32, f32) {
    let clamped = pan.clamp(-1.0, 1.0);
    let angle = (clamped + 1.0) * 0.5 * std::f32::consts::FRAC_PI_2;
    (angle.cos(), angle.sin())
}

// ---------------------------------------------------------------------------
// Stereo chorus config
// ---------------------------------------------------------------------------

/// Configuration for stereo imaging of a multi-voice chorus.
///
/// Controls per-voice stereo positions, overall stereo width, and
/// mono compatibility settings. Used by [`apply_stereo_mix`] to
/// produce a stereo (L, R) pair from mono voice buffers.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StereoChorusConfig {
    /// Per-voice stereo positions.
    pub positions: Vec<StereoPosition>,

    /// Stereo width multiplier in [0.0, 1.0].
    ///
    /// Scales all pan positions toward center. At 0.0 all voices are
    /// centered (mono output). At 1.0 voices use their full positions.
    pub stereo_width: f32,

    /// When true, apply `1/sqrt(2)` normalization per voice to ensure
    /// that `L + R` approximates the mono mix amplitude at all pan positions.
    ///
    /// Without this, center-panned voices are louder in mono folddown
    /// (`cos(pi/4) + sin(pi/4) = sqrt(2) ~ 1.414`) compared to hard-panned
    /// voices (`1.0 + 0.0 = 1.0`).
    pub mono_compatible: bool,
}

impl StereoChorusConfig {
    /// Create a config with the given positions and default settings.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any custom pan position is
    /// non-finite.
    pub fn new(positions: Vec<StereoPosition>) -> Result<Self, KokoroError> {
        for (i, pos) in positions.iter().enumerate() {
            if let StereoPosition::Custom(p) = pos {
                if !p.is_finite() {
                    return Err(KokoroError::InvalidConfig {
                        field: "positions",
                        reason: format!("position[{i}] = Custom({p}): must be finite"),
                    });
                }
            }
        }
        Ok(Self {
            positions,
            stereo_width: 1.0,
            mono_compatible: false,
        })
    }

    /// Create a config with the default layout for the given voice count.
    ///
    /// Uses [`default_voice_layout`] to assign positions automatically.
    pub fn auto_layout(n_voices: usize) -> Result<Self, KokoroError> {
        if n_voices == 0 || n_voices > 32 {
            return Err(KokoroError::InvalidConfig {
                field: "n_voices",
                reason: format!("must be 1..=32, got {n_voices}"),
            });
        }
        Self::new(default_voice_layout(n_voices))
    }

    /// Set stereo width (0.0 = mono, 1.0 = full stereo).
    #[must_use]
    pub fn with_stereo_width(mut self, width: f32) -> Self {
        self.stereo_width = width.clamp(0.0, 1.0);
        self
    }

    /// Enable mono-compatible normalization.
    #[must_use]
    pub fn with_mono_compatible(mut self, enabled: bool) -> Self {
        self.mono_compatible = enabled;
        self
    }

    /// Validate the config.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if !self.stereo_width.is_finite() || !(0.0..=1.0).contains(&self.stereo_width) {
            return Err(KokoroError::InvalidConfig {
                field: "stereo_width",
                reason: format!(
                    "stereo_width = {}: must be finite and in [0.0, 1.0]",
                    self.stereo_width,
                ),
            });
        }
        for (i, pos) in self.positions.iter().enumerate() {
            if let StereoPosition::Custom(p) = pos {
                if !p.is_finite() || !(-1.0..=1.0).contains(p) {
                    return Err(KokoroError::InvalidConfig {
                        field: "positions",
                        reason: format!(
                            "position[{i}] = Custom({p}): must be finite and in [-1.0, 1.0]"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Get the effective pan values after applying stereo width.
    #[must_use]
    pub fn effective_pans(&self) -> Vec<f32> {
        self.positions
            .iter()
            .map(|pos| (pos.to_pan() * self.stereo_width).clamp(-1.0, 1.0))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Stereo mix function
// ---------------------------------------------------------------------------

/// Apply stereo imaging to mono voice buffers, returning separate L/R channels.
///
/// Uses constant-power (sin/cos) pan law to position each voice in the
/// stereo field according to the config. Each voice's mono PCM is split
/// into left and right contributions based on its pan position.
///
/// # Returns
///
/// `(left_channel, right_channel)` — each with the same length as the
/// longest voice buffer. Shorter voices are zero-padded.
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if `voices` length does not match
/// the number of positions in `config`.
pub fn apply_stereo_mix(
    voices: &[Vec<f32>],
    config: &StereoChorusConfig,
) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
    apply_stereo_mix_refs(
        &voices.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        config,
    )
}

/// Apply stereo imaging to borrowed voice slices (ref-slice variant).
///
/// Same as [`apply_stereo_mix`] but accepts borrowed slices to avoid
/// cloning PCM data.
pub fn apply_stereo_mix_refs(
    voices: &[&[f32]],
    config: &StereoChorusConfig,
) -> Result<(Vec<f32>, Vec<f32>), KokoroError> {
    config.validate()?;
    if voices.len() != config.positions.len() {
        return Err(KokoroError::InvalidInput(format!(
            "voices length {} != positions length {}",
            voices.len(),
            config.positions.len(),
        )));
    }
    if voices.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let max_len = voices.iter().map(|v| v.len()).max().unwrap_or(0);
    if max_len == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let panner = StereoPanner::new(config.stereo_width);

    // Mono compatibility normalization factor: 1/sqrt(2) ensures that
    // L + R ~ 1.0 at center pan (where cos + sin = sqrt(2)).
    let mono_norm = if config.mono_compatible {
        std::f32::consts::FRAC_1_SQRT_2
    } else {
        1.0
    };

    let mut left = vec![0.0f32; max_len];
    let mut right = vec![0.0f32; max_len];

    for (pcm, pos) in voices.iter().zip(config.positions.iter()) {
        let pan = pos.to_pan();
        let (lg, rg) = panner.pan_gains(pan);
        let left_gain = lg * mono_norm;
        let right_gain = rg * mono_norm;

        for (i, &sample) in pcm.iter().enumerate() {
            left[i] += sample * left_gain;
            right[i] += sample * right_gain;
        }
    }

    Ok((left, right))
}

/// Convert separate L/R channels to interleaved stereo format.
///
/// Output: `[L0, R0, L1, R1, ...]`. Both channels must have the same length.
///
/// # Errors
///
/// Returns `KokoroError::InvalidInput` if `left` and `right` have different
/// lengths.
pub fn interleave_stereo(left: &[f32], right: &[f32]) -> Result<Vec<f32>, KokoroError> {
    if left.len() != right.len() {
        return Err(KokoroError::InvalidInput(format!(
            "left length {} != right length {}",
            left.len(),
            right.len(),
        )));
    }
    let mut interleaved = Vec::with_capacity(left.len() * 2);
    for (&l, &r) in left.iter().zip(right.iter()) {
        interleaved.push(l);
        interleaved.push(r);
    }
    Ok(interleaved)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_stereo_tests.rs"]
mod tests;

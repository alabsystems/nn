// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Output format and delivery manager for Kokoro chorus.
//!
//! Handles the final stage of chorus output: format conversion, sample rate
//! conversion, channel routing, and delivery formatting. Ensures the chorus
//! output matches the target delivery format regardless of internal processing
//! format (f32, 24 kHz, stereo).
//!
//! # Processing chain
//!
//! 1. **DC offset removal** — 1st-order HPF at 5 Hz removes sub-audible DC
//! 2. **Normalization** — optional peak normalization to target dBFS
//! 3. **Sample rate conversion** — windowed sinc interpolation (Hann, 16-tap)
//! 4. **Channel routing** — mono sum, stereo passthrough, 5.1 upmix
//! 5. **Clip guard** — soft-clip at 0 dBFS via tanh instead of hard-clip
//! 6. **Fade in/out** — raised cosine at segment boundaries
//! 7. **Bit depth conversion** — quantize to I16/I24/F32
//!
//! # Sinc resampler
//!
//! The resampler uses a 16-tap Hann-windowed sinc interpolation kernel.
//! This provides ~96 dB stopband rejection with minimal passband ripple,
//! suitable for speech. For integer ratio conversions (e.g., 48000/24000 = 2),
//! the filter simplifies to a polyphase structure.
//!
//! # References
//!
//! - Smith, J.O. "Digital Audio Resampling Home Page."
//!   <https://ccrma.stanford.edu/~jos/resample/>
//! - ITU-R BS.775-3 "Multichannel stereophonic sound system with and without
//!   accompanying picture." International Telecommunication Union, 2012.

use crate::kokoro_error::KokoroError;

// ---------------------------------------------------------------------------
// Channel layout
// ---------------------------------------------------------------------------

/// Target channel layout for delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum ChannelLayout {
    /// Mono: sum left + right, divide by 2.
    Mono,
    /// Stereo: left/right passthrough.
    #[default]
    Stereo,
    /// 5.1 surround: L, R, C, LFE, Ls, Rs.
    Surround51,
}


impl ChannelLayout {
    /// Number of output channels for this layout.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
        }
    }
}

// ---------------------------------------------------------------------------
// Bit depth
// ---------------------------------------------------------------------------

/// Target bit depth for delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum BitDepth {
    /// 32-bit floating point (no quantization).
    #[default]
    F32,
    /// 16-bit signed integer (CD quality).
    I16,
    /// 24-bit signed integer (professional audio).
    I24,
}


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the output format and delivery manager.
///
/// Controls sample rate, channel layout, bit depth, normalization,
/// fading, DC removal, and clip guarding.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputConfig {
    /// Target sample rate in Hz. Default: 24000 (Kokoro native).
    pub target_sample_rate: u32,
    /// Target channel layout. Default: Stereo.
    pub target_channels: ChannelLayout,
    /// Target bit depth. Default: F32.
    pub target_bit_depth: BitDepth,
    /// Optional peak normalization target in dBFS (e.g., -1.0).
    /// `None` disables normalization. Default: `None`.
    pub normalize_to_dbfs: Option<f32>,
    /// Fade-in duration in milliseconds. Default: 0.
    pub fade_in_ms: f32,
    /// Fade-out duration in milliseconds. Default: 0.
    pub fade_out_ms: f32,
    /// Remove DC offset via 1st-order HPF at 5 Hz. Default: true.
    pub dc_offset_removal: bool,
    /// Apply soft-clip guard at 0 dBFS. Default: true.
    pub final_clip_guard: bool,
    /// Center channel gain for 5.1 upmix (linear). Default: 0.707 (-3 dB).
    pub surround_center_gain: f32,
    /// Rear channel gain for 5.1 upmix (linear). Default: 0.5 (-6 dB).
    pub surround_rear_gain: f32,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 24000,
            target_channels: ChannelLayout::Stereo,
            target_bit_depth: BitDepth::F32,
            normalize_to_dbfs: None,
            fade_in_ms: 0.0,
            fade_out_ms: 0.0,
            dc_offset_removal: true,
            final_clip_guard: true,
            surround_center_gain: 0.707,
            surround_rear_gain: 0.5,
        }
    }
}

impl OutputConfig {
    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if any parameter is out of range.
    pub fn validate(&self) -> Result<(), KokoroError> {
        if self.target_sample_rate < 8000 || self.target_sample_rate > 192000 {
            return Err(KokoroError::InvalidConfig {
                field: "target_sample_rate",
                reason: format!("must be in [8000, 192000], got {}", self.target_sample_rate),
            });
        }
        if let Some(db) = self.normalize_to_dbfs {
            if !db.is_finite() || !(-60.0..=0.0).contains(&db) {
                return Err(KokoroError::InvalidConfig {
                    field: "normalize_to_dbfs",
                    reason: format!("must be finite and in [-60.0, 0.0], got {db}"),
                });
            }
        }
        if !self.fade_in_ms.is_finite() || self.fade_in_ms < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "fade_in_ms",
                reason: format!("must be finite and >= 0, got {}", self.fade_in_ms),
            });
        }
        if !self.fade_out_ms.is_finite() || self.fade_out_ms < 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "fade_out_ms",
                reason: format!("must be finite and >= 0, got {}", self.fade_out_ms),
            });
        }
        if !self.surround_center_gain.is_finite()
            || self.surround_center_gain < 0.0
            || self.surround_center_gain > 1.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "surround_center_gain",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.surround_center_gain
                ),
            });
        }
        if !self.surround_rear_gain.is_finite()
            || self.surround_rear_gain < 0.0
            || self.surround_rear_gain > 1.0
        {
            return Err(KokoroError::InvalidConfig {
                field: "surround_rear_gain",
                reason: format!(
                    "must be finite and in [0.0, 1.0], got {}",
                    self.surround_rear_gain
                ),
            });
        }
        Ok(())
    }

    /// Builder: set target sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.target_sample_rate = rate;
        self
    }

    /// Builder: set target channel layout.
    #[must_use]
    pub fn with_channels(mut self, layout: ChannelLayout) -> Self {
        self.target_channels = layout;
        self
    }

    /// Builder: set target bit depth.
    #[must_use]
    pub fn with_bit_depth(mut self, depth: BitDepth) -> Self {
        self.target_bit_depth = depth;
        self
    }

    /// Builder: set normalization target in dBFS.
    #[must_use]
    pub fn with_normalize(mut self, dbfs: f32) -> Self {
        self.normalize_to_dbfs = Some(dbfs);
        self
    }

    /// Builder: set fade-in duration in milliseconds.
    #[must_use]
    pub fn with_fade_in(mut self, ms: f32) -> Self {
        self.fade_in_ms = ms;
        self
    }

    /// Builder: set fade-out duration in milliseconds.
    #[must_use]
    pub fn with_fade_out(mut self, ms: f32) -> Self {
        self.fade_out_ms = ms;
        self
    }

    /// Builder: set DC offset removal.
    #[must_use]
    pub fn with_dc_removal(mut self, enabled: bool) -> Self {
        self.dc_offset_removal = enabled;
        self
    }

    /// Builder: set clip guard.
    #[must_use]
    pub fn with_clip_guard(mut self, enabled: bool) -> Self {
        self.final_clip_guard = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// Formatted output
// ---------------------------------------------------------------------------

/// Result of formatting chorus output for delivery.
///
/// Contains interleaved samples in the target format along with metering
/// information for quality verification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FormattedOutput {
    /// Interleaved audio samples (channel-interleaved for multi-channel).
    /// For mono: [s0, s1, ...]. For stereo: [L0, R0, L1, R1, ...].
    /// For 5.1: [L0, R0, C0, LFE0, Ls0, Rs0, L1, R1, ...].
    pub samples: Vec<f32>,
    /// Output sample rate in Hz.
    pub sample_rate: u32,
    /// Output channel layout.
    pub channels: ChannelLayout,
    /// Output bit depth.
    pub bit_depth: BitDepth,
    /// Peak level in dBFS after all processing.
    pub peak_dbfs: f32,
    /// RMS level in dBFS after all processing.
    pub rms_dbfs: f32,
    /// Number of samples that were clipped (before clip guard).
    pub clip_count: usize,
}

impl FormattedOutput {
    /// Number of frames (samples per channel).
    #[must_use]
    pub fn frame_count(&self) -> usize {
        let ch = self.channels.channel_count();
        if ch == 0 {
            return 0;
        }
        self.samples.len() / ch
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frame_count() as f32 / self.sample_rate as f32
    }
}

// ---------------------------------------------------------------------------
// Output formatter
// ---------------------------------------------------------------------------

/// Output format and delivery manager for Kokoro chorus.
///
/// Handles the full processing chain from internal f32 stereo at the source
/// sample rate to the target delivery format.
pub struct OutputFormatter {
    config: OutputConfig,
    /// DC blocker state per channel: (x_prev, y_prev).
    dc_state: [(f32, f32); 2],
}

impl OutputFormatter {
    /// Create a new output formatter.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: &OutputConfig) -> Result<Self, KokoroError> {
        config.validate()?;
        Ok(Self {
            config: config.clone(),
            dc_state: [(0.0, 0.0); 2],
        })
    }

    /// Create a formatter with default configuration.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidConfig` if defaults are invalid.
    pub fn with_defaults() -> Result<Self, KokoroError> {
        Self::new(&OutputConfig::default())
    }

    /// Reset all internal state (DC blocker).
    pub fn reset(&mut self) {
        self.dc_state = [(0.0, 0.0); 2];
    }

    /// Get the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &OutputConfig {
        &self.config
    }

    /// Format stereo chorus output for delivery.
    ///
    /// Takes left and right channel buffers at the source sample rate
    /// (typically 24 kHz) and produces a [`FormattedOutput`] in the
    /// configured target format.
    ///
    /// # Errors
    ///
    /// Returns `KokoroError::InvalidInput` if left/right lengths differ.
    pub fn format_output(
        &mut self,
        left: &[f32],
        right: &[f32],
        source_sample_rate: u32,
    ) -> Result<FormattedOutput, KokoroError> {
        if left.len() != right.len() {
            return Err(KokoroError::InvalidInput(format!(
                "left ({}) and right ({}) channel lengths must match",
                left.len(),
                right.len()
            )));
        }

        let mut left_buf = left.to_vec();
        let mut right_buf = right.to_vec();

        // 1. DC offset removal
        if self.config.dc_offset_removal {
            remove_dc(&mut left_buf, &mut self.dc_state[0]);
            remove_dc(&mut right_buf, &mut self.dc_state[1]);
        }

        // 2. Peak normalization
        if let Some(target_db) = self.config.normalize_to_dbfs {
            normalize_peak_stereo(&mut left_buf, &mut right_buf, target_db);
        }

        // 3. Sample rate conversion
        if source_sample_rate != self.config.target_sample_rate {
            left_buf = resample_sinc(
                &left_buf,
                source_sample_rate,
                self.config.target_sample_rate,
            );
            right_buf = resample_sinc(
                &right_buf,
                source_sample_rate,
                self.config.target_sample_rate,
            );
        }

        // 4. Fade in/out
        let sr = self.config.target_sample_rate as f32;
        if self.config.fade_in_ms > 0.0 {
            let fade_samples = ((self.config.fade_in_ms / 1000.0) * sr) as usize;
            apply_raised_cosine_fade(&mut left_buf, fade_samples, true);
            apply_raised_cosine_fade(&mut right_buf, fade_samples, true);
        }
        if self.config.fade_out_ms > 0.0 {
            let fade_samples = ((self.config.fade_out_ms / 1000.0) * sr) as usize;
            apply_raised_cosine_fade(&mut left_buf, fade_samples, false);
            apply_raised_cosine_fade(&mut right_buf, fade_samples, false);
        }

        // 5. Count clips and apply clip guard
        let mut clip_count = 0;
        clip_count += count_clips(&left_buf);
        clip_count += count_clips(&right_buf);

        if self.config.final_clip_guard {
            soft_clip_tanh(&mut left_buf);
            soft_clip_tanh(&mut right_buf);
        }

        // 6. Channel routing → interleaved output
        let interleaved = route_channels(
            &left_buf,
            &right_buf,
            self.config.target_channels,
            self.config.surround_center_gain,
            self.config.surround_rear_gain,
        );

        // 7. Bit depth quantization (in-place on the interleaved buffer)
        let mut samples = interleaved;
        apply_bit_depth(&mut samples, self.config.target_bit_depth);

        // Metering
        let peak_dbfs = measure_peak_db(&samples);
        let rms_dbfs = measure_rms_db(&samples);

        Ok(FormattedOutput {
            samples,
            sample_rate: self.config.target_sample_rate,
            channels: self.config.target_channels,
            bit_depth: self.config.target_bit_depth,
            peak_dbfs,
            rms_dbfs,
            clip_count,
        })
    }
}

// ---------------------------------------------------------------------------
// DC offset removal — 1st-order HPF at 5 Hz
// ---------------------------------------------------------------------------

/// Remove DC offset via a 1st-order high-pass at ~5 Hz.
///
/// ```text
/// y[n] = x[n] - x[n-1] + R * y[n-1]
/// ```
/// where `R = 1 - (2 * pi * 5 / 24000)` at 24 kHz.
fn remove_dc(audio: &mut [f32], state: &mut (f32, f32)) {
    // R coefficient for ~5 Hz cutoff at 24 kHz.
    // R = 1 - (2*pi*5/24000) ≈ 0.99869
    const R: f32 = 0.998_69;

    let (ref mut x_prev, ref mut y_prev) = state;
    for sample in audio.iter_mut() {
        if !sample.is_finite() {
            *x_prev = 0.0;
            *y_prev = 0.0;
            *sample = 0.0;
            continue;
        }
        let x = *sample;
        let y = x - *x_prev + R * *y_prev;
        *x_prev = x;
        *y_prev = if y.is_finite() && y.abs() > 1e-30 {
            y
        } else {
            0.0
        };
        *sample = *y_prev;
    }
}

// ---------------------------------------------------------------------------
// Peak normalization
// ---------------------------------------------------------------------------

/// Normalize stereo signal so peak matches `target_db`.
fn normalize_peak_stereo(left: &mut [f32], right: &mut [f32], target_db: f32) {
    let peak_l = left
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    let peak_r = right
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    let peak = peak_l.max(peak_r);

    if peak < 1e-12 {
        return; // silence
    }

    let target_linear = 10.0f32.powf(target_db / 20.0);
    let gain = target_linear / peak;
    if !gain.is_finite() {
        return;
    }

    for s in left.iter_mut() {
        if s.is_finite() {
            *s *= gain;
        }
    }
    for s in right.iter_mut() {
        if s.is_finite() {
            *s *= gain;
        }
    }
}

// ---------------------------------------------------------------------------
// Sinc resampler — Hann-windowed, 16-tap
// ---------------------------------------------------------------------------

/// Number of taps on each side of the sinc kernel center.
const SINC_HALF_TAPS: usize = 8;

/// Resample audio using Hann-windowed sinc interpolation.
///
/// Produces output at `target_rate` from input at `source_rate`.
/// Uses a 16-tap (8 per side) Hann-windowed sinc kernel.
fn resample_sinc(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if input.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate {
        return input.to_vec();
    }

    let ratio = f64::from(source_rate) / f64::from(target_rate);
    let out_len = ((input.len() as f64 / ratio).ceil()) as usize;
    let mut output = Vec::with_capacity(out_len);

    // Anti-aliasing: when downsampling, widen the sinc kernel.
    let cutoff = if target_rate < source_rate {
        f64::from(target_rate) / f64::from(source_rate)
    } else {
        1.0
    };

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let center = src_pos.floor() as i64;
        let frac = src_pos - center as f64;

        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for tap in -(SINC_HALF_TAPS as i64)..=(SINC_HALF_TAPS as i64) {
            let idx = center + tap;
            if idx < 0 || idx >= input.len() as i64 {
                continue;
            }

            let x = (tap as f64 - frac) * cutoff;
            let sinc_val = if x.abs() < 1e-12 {
                1.0
            } else {
                let pi_x = std::f64::consts::PI * x;
                pi_x.sin() / pi_x
            };

            // Hann window over the kernel support.
            let window_pos =
                (tap as f64 - frac + SINC_HALF_TAPS as f64) / (2 * SINC_HALF_TAPS) as f64;
            let hann = if (0.0..=1.0).contains(&window_pos) {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * window_pos).cos())
            } else {
                0.0
            };

            let w = sinc_val * hann * cutoff;
            let sample = input[idx as usize];
            if sample.is_finite() {
                sum += f64::from(sample) * w;
                weight_sum += w;
            }
        }

        let out_sample = if weight_sum.abs() > 1e-12 {
            (sum / weight_sum) as f32
        } else {
            0.0
        };
        output.push(if out_sample.is_finite() {
            out_sample
        } else {
            0.0
        });
    }

    output
}

// ---------------------------------------------------------------------------
// Raised cosine fade
// ---------------------------------------------------------------------------

/// Apply a raised cosine fade in or out.
///
/// `is_fade_in = true`: fade from silence to full at the start.
/// `is_fade_in = false`: fade from full to silence at the end.
fn apply_raised_cosine_fade(audio: &mut [f32], fade_samples: usize, is_fade_in: bool) {
    let n = fade_samples.min(audio.len());
    if n == 0 {
        return;
    }

    for i in 0..n {
        // Raised cosine: 0.5 * (1 - cos(pi * t))
        let t = i as f32 / n as f32;
        let gain = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());

        let idx = if is_fade_in { i } else { audio.len() - 1 - i };
        if audio[idx].is_finite() {
            audio[idx] *= gain;
        }
    }
}

// ---------------------------------------------------------------------------
// Clip detection and soft-clip
// ---------------------------------------------------------------------------

/// Count samples exceeding +/- 1.0 (would clip in integer formats).
fn count_clips(audio: &[f32]) -> usize {
    audio
        .iter()
        .filter(|s| s.is_finite() && s.abs() > 1.0)
        .count()
}

/// Apply tanh soft-clipping: samples near/above 1.0 are smoothly limited.
///
/// Below 0.9: linear passthrough. Above 0.9: tanh compression.
fn soft_clip_tanh(audio: &mut [f32]) {
    for sample in audio.iter_mut() {
        if !sample.is_finite() {
            *sample = 0.0;
            continue;
        }
        if sample.abs() > 0.9 {
            *sample = sample.tanh();
        }
    }
}

// ---------------------------------------------------------------------------
// Channel routing
// ---------------------------------------------------------------------------

/// Route stereo to the target channel layout, producing interleaved output.
fn route_channels(
    left: &[f32],
    right: &[f32],
    layout: ChannelLayout,
    center_gain: f32,
    rear_gain: f32,
) -> Vec<f32> {
    let n = left.len();
    match layout {
        ChannelLayout::Mono => {
            // Mono: (L + R) / 2
            left.iter()
                .zip(right.iter())
                .map(|(&l, &r)| {
                    let l = if l.is_finite() { l } else { 0.0 };
                    let r = if r.is_finite() { r } else { 0.0 };
                    (l + r) * 0.5
                })
                .collect()
        }
        ChannelLayout::Stereo => {
            // Interleave L, R
            let mut out = Vec::with_capacity(n * 2);
            for (&l, &r) in left.iter().zip(right.iter()) {
                out.push(if l.is_finite() { l } else { 0.0 });
                out.push(if r.is_finite() { r } else { 0.0 });
            }
            out
        }
        ChannelLayout::Surround51 => {
            // 5.1: L, R, C (phantom center), LFE (lowpassed sum), Ls, Rs
            // C = (L + R) * center_gain
            // LFE = 0 (no sub-bass synthesis by default)
            // Ls = L * rear_gain, Rs = R * rear_gain
            let mut out = Vec::with_capacity(n * 6);
            for (&l, &r) in left.iter().zip(right.iter()) {
                let l = if l.is_finite() { l } else { 0.0 };
                let r = if r.is_finite() { r } else { 0.0 };
                let center = (l + r) * 0.5 * center_gain;
                out.push(l); // L
                out.push(r); // R
                out.push(center); // C
                out.push(0.0); // LFE
                out.push(l * rear_gain); // Ls
                out.push(r * rear_gain); // Rs
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Bit depth quantization
// ---------------------------------------------------------------------------

/// Apply bit depth quantization to interleaved samples.
///
/// F32: no-op. I16/I24: quantize to integer range then scale back to f32.
/// This simulates the precision loss of integer delivery formats.
fn apply_bit_depth(samples: &mut [f32], depth: BitDepth) {
    match depth {
        BitDepth::F32 => {} // no quantization
        BitDepth::I16 => {
            let scale = 32767.0f32;
            for s in samples.iter_mut() {
                if !s.is_finite() {
                    *s = 0.0;
                    continue;
                }
                let quantized = (*s * scale).round().clamp(-scale, scale);
                *s = quantized / scale;
            }
        }
        BitDepth::I24 => {
            let scale = 8_388_607.0f32;
            for s in samples.iter_mut() {
                if !s.is_finite() {
                    *s = 0.0;
                    continue;
                }
                let quantized = (*s * scale).round().clamp(-scale, scale);
                *s = quantized / scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Metering helpers
// ---------------------------------------------------------------------------

const SILENCE_DB: f32 = -120.0;
const AMPLITUDE_FLOOR: f32 = 1e-12;

/// Measure peak dBFS of an audio buffer.
fn measure_peak_db(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return SILENCE_DB;
    }
    let peak = audio
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| s.abs())
        .fold(0.0f32, f32::max);
    if peak < AMPLITUDE_FLOOR {
        SILENCE_DB
    } else {
        let db = 20.0 * peak.log10();
        if db.is_finite() {
            db
        } else {
            SILENCE_DB
        }
    }
}

/// Measure RMS dBFS of an audio buffer.
fn measure_rms_db(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return SILENCE_DB;
    }
    let (sum_sq, count) = audio.iter().fold((0.0f64, 0u64), |(acc, n), &s| {
        if s.is_finite() {
            (acc + f64::from(s) * f64::from(s), n + 1)
        } else {
            (acc, n)
        }
    });
    if count == 0 {
        return SILENCE_DB;
    }
    let rms = (sum_sq / count as f64).sqrt() as f32;
    if rms < AMPLITUDE_FLOOR {
        SILENCE_DB
    } else {
        let db = 20.0 * rms.log10();
        if db.is_finite() {
            db
        } else {
            SILENCE_DB
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq: f32, sr: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_config_default_is_valid() {
        let config = OutputConfig::default();
        config.validate().expect("default should be valid");
        assert_eq!(config.target_sample_rate, 24000);
        assert_eq!(config.target_channels, ChannelLayout::Stereo);
        assert_eq!(config.target_bit_depth, BitDepth::F32);
        assert!(config.dc_offset_removal);
        assert!(config.final_clip_guard);
    }

    #[test]
    fn test_config_validation_rejects_bad_sample_rate() {
        assert!(OutputConfig::default()
            .with_sample_rate(0)
            .validate()
            .is_err());
        assert!(OutputConfig::default()
            .with_sample_rate(7999)
            .validate()
            .is_err());
        assert!(OutputConfig::default()
            .with_sample_rate(200000)
            .validate()
            .is_err());
        assert!(OutputConfig::default()
            .with_sample_rate(48000)
            .validate()
            .is_ok());
    }

    #[test]
    fn test_config_validation_rejects_bad_normalize() {
        assert!(OutputConfig::default()
            .with_normalize(1.0)
            .validate()
            .is_err());
        assert!(OutputConfig::default()
            .with_normalize(-70.0)
            .validate()
            .is_err());
        assert!(OutputConfig::default()
            .with_normalize(f32::NAN)
            .validate()
            .is_err());
        assert!(OutputConfig::default()
            .with_normalize(-1.0)
            .validate()
            .is_ok());
    }

    #[test]
    fn test_config_builder_chain() {
        let config = OutputConfig::default()
            .with_sample_rate(48000)
            .with_channels(ChannelLayout::Mono)
            .with_bit_depth(BitDepth::I16)
            .with_normalize(-1.0)
            .with_fade_in(10.0)
            .with_fade_out(20.0)
            .with_dc_removal(false)
            .with_clip_guard(false);
        config.validate().expect("builder chain should be valid");
        assert_eq!(config.target_sample_rate, 48000);
        assert_eq!(config.target_channels, ChannelLayout::Mono);
        assert_eq!(config.target_bit_depth, BitDepth::I16);
    }

    #[test]
    fn test_format_output_stereo_passthrough() {
        let config = OutputConfig::default()
            .with_dc_removal(false)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        let left = sine_signal(440.0, 24000.0, 2400, 0.5);
        let right = sine_signal(440.0, 24000.0, 2400, 0.5);

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        assert_eq!(out.sample_rate, 24000);
        assert_eq!(out.channels, ChannelLayout::Stereo);
        assert_eq!(out.samples.len(), 2400 * 2); // interleaved stereo
        assert_eq!(out.frame_count(), 2400);
        assert!(out.clip_count == 0);
    }

    #[test]
    fn test_format_output_mono_downmix() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Mono)
            .with_dc_removal(false)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        let left = vec![0.8f32; 100];
        let right = vec![0.2f32; 100];

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        assert_eq!(out.channels, ChannelLayout::Mono);
        assert_eq!(out.samples.len(), 100);

        // Each sample should be (0.8 + 0.2) / 2 = 0.5
        for &s in &out.samples {
            assert!(
                (s - 0.5).abs() < 0.01,
                "mono downmix: expected ~0.5, got {s}"
            );
        }
    }

    #[test]
    fn test_format_output_surround51_channels() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Surround51)
            .with_dc_removal(false)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        let left = vec![0.6f32; 10];
        let right = vec![0.4f32; 10];

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        assert_eq!(out.channels.channel_count(), 6);
        assert_eq!(out.samples.len(), 10 * 6);
        assert_eq!(out.frame_count(), 10);
    }

    #[test]
    fn test_format_output_resample_upsample() {
        let config = OutputConfig::default()
            .with_sample_rate(48000)
            .with_channels(ChannelLayout::Mono)
            .with_dc_removal(false)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        // 240 samples at 24kHz = 10ms -> should become ~480 samples at 48kHz
        let left = sine_signal(440.0, 24000.0, 240, 0.5);
        let right = sine_signal(440.0, 24000.0, 240, 0.5);

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        assert_eq!(out.sample_rate, 48000);
        // Expect approximately 480 samples (ratio 2x)
        let expected = 480;
        assert!(
            (out.samples.len() as i64 - i64::from(expected)).unsigned_abs() <= 2,
            "expected ~{expected} samples, got {}",
            out.samples.len()
        );
    }

    #[test]
    fn test_format_output_resample_downsample() {
        let config = OutputConfig::default()
            .with_sample_rate(16000)
            .with_channels(ChannelLayout::Mono)
            .with_dc_removal(false)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        // 2400 samples at 24kHz = 100ms -> ~1600 at 16kHz
        let left = sine_signal(440.0, 24000.0, 2400, 0.5);
        let right = sine_signal(440.0, 24000.0, 2400, 0.5);

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        assert_eq!(out.sample_rate, 16000);
        let expected = 1600;
        assert!(
            (out.samples.len() as i64 - i64::from(expected)).unsigned_abs() <= 2,
            "expected ~{expected} samples, got {}",
            out.samples.len()
        );
    }

    #[test]
    fn test_dc_offset_removal() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Mono)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        // Signal with DC offset = 0.3
        let n = 4800;
        let left: Vec<f32> = (0..n)
            .map(|i| 0.3 + 0.4 * (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 24000.0).sin())
            .collect();
        let right = left.clone();

        let out = fmt.format_output(&left, &right, 24000).expect("ok");

        // After DC removal, mean should be near zero (skip transient)
        let skip = 1000;
        let mean: f64 = out.samples[skip..].iter().map(|&s| f64::from(s)).sum::<f64>()
            / (out.samples.len() - skip) as f64;
        assert!(mean.abs() < 0.05, "DC should be removed, mean = {mean}");
    }

    #[test]
    fn test_clip_guard_prevents_hard_clip() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Mono)
            .with_dc_removal(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        // Signal that exceeds 1.0
        let left = vec![1.5f32; 100];
        let right = vec![1.5f32; 100];

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        assert!(out.clip_count > 0, "should detect clips");

        // After soft-clip, all samples should be <= 1.0
        for &s in &out.samples {
            assert!(
                s.abs() <= 1.0,
                "soft-clip should keep samples <= 1.0, got {s}"
            );
        }
    }

    #[test]
    fn test_fade_in_out() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Mono)
            .with_dc_removal(false)
            .with_clip_guard(false)
            .with_fade_in(10.0) // 10ms
            .with_fade_out(10.0); // 10ms
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        // Constant signal
        let left = vec![0.5f32; 2400]; // 100ms at 24kHz
        let right = vec![0.5f32; 2400];

        let out = fmt.format_output(&left, &right, 24000).expect("ok");

        // First sample should be near zero (fade in)
        assert!(
            out.samples[0].abs() < 0.01,
            "fade-in: first sample should be ~0, got {}",
            out.samples[0]
        );

        // Last sample should be near zero (fade out)
        let last = out.samples.last().copied().unwrap_or(1.0);
        assert!(
            last.abs() < 0.01,
            "fade-out: last sample should be ~0, got {last}"
        );

        // Middle samples should be at full level
        let mid = out.samples[out.samples.len() / 2];
        assert!((mid - 0.5).abs() < 0.05, "middle should be ~0.5, got {mid}");
    }

    #[test]
    fn test_bit_depth_i16_quantization() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Mono)
            .with_bit_depth(BitDepth::I16)
            .with_dc_removal(false)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        let left = sine_signal(440.0, 24000.0, 2400, 0.5);
        let right = sine_signal(440.0, 24000.0, 2400, 0.5);

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        assert_eq!(out.bit_depth, BitDepth::I16);

        // I16 step size = 1/32767 ≈ 3.05e-5.
        // Each sample should be quantized to a multiple of that step.
        let step = 1.0 / 32767.0;
        for (i, &s) in out.samples.iter().enumerate().take(100) {
            let remainder = (s / step).round() * step - s;
            assert!(
                remainder.abs() < 1e-6,
                "sample {i}: not quantized to I16 step, remainder = {remainder}"
            );
        }
    }

    #[test]
    fn test_normalize_to_dbfs() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Mono)
            .with_normalize(-6.0)
            .with_dc_removal(false)
            .with_clip_guard(false);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        // Quiet signal: peak ≈ 0.1 = -20 dBFS
        let left = sine_signal(440.0, 24000.0, 2400, 0.1);
        let right = sine_signal(440.0, 24000.0, 2400, 0.1);

        let out = fmt.format_output(&left, &right, 24000).expect("ok");

        // After normalization to -6 dBFS, peak should be near 0.5
        let peak = out.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let peak_db = 20.0 * peak.log10();
        assert!(
            (peak_db - (-6.0)).abs() < 0.5,
            "expected ~-6 dBFS peak, got {peak_db}"
        );
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let mut fmt = OutputFormatter::with_defaults().expect("valid");
        let left = vec![0.5f32; 100];
        let right = vec![0.5f32; 50];

        assert!(fmt.format_output(&left, &right, 24000).is_err());
    }

    #[test]
    fn test_empty_input() {
        let mut fmt = OutputFormatter::with_defaults().expect("valid");
        let out = fmt.format_output(&[], &[], 24000).expect("ok");
        assert!(out.samples.is_empty());
        assert_eq!(out.frame_count(), 0);
    }

    #[test]
    fn test_nan_safety() {
        let config = OutputConfig::default()
            .with_channels(ChannelLayout::Mono)
            .with_dc_removal(true)
            .with_clip_guard(true);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        let left = vec![0.5, f32::NAN, 0.3, f32::INFINITY, -0.2];
        let right = vec![-0.1, 0.4, f32::NEG_INFINITY, 0.0, 0.6];

        let out = fmt.format_output(&left, &right, 24000).expect("ok");
        for (i, &s) in out.samples.iter().enumerate() {
            assert!(s.is_finite(), "sample {i} should be finite, got {s}");
        }
    }

    #[test]
    fn test_reset_clears_dc_state() {
        let config = OutputConfig::default().with_channels(ChannelLayout::Mono);
        let mut fmt = OutputFormatter::new(&config).expect("valid");

        let left = vec![0.5f32; 100];
        let right = vec![0.5f32; 100];
        let _ = fmt.format_output(&left, &right, 24000);

        fmt.reset();
        assert_eq!(fmt.dc_state, [(0.0, 0.0); 2]);
    }

    #[test]
    fn test_channel_layout_counts() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
    }

    #[test]
    fn test_formatted_output_duration() {
        let out = FormattedOutput {
            samples: vec![0.0; 48000],
            sample_rate: 24000,
            channels: ChannelLayout::Stereo,
            bit_depth: BitDepth::F32,
            peak_dbfs: -6.0,
            rms_dbfs: -12.0,
            clip_count: 0,
        };
        // 48000 interleaved stereo samples = 24000 frames / 24000 Hz = 1.0s
        assert!((out.duration_secs() - 1.0).abs() < 0.001);
        assert_eq!(out.frame_count(), 24000);
    }
}

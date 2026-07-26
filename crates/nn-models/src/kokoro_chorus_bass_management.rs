// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Psychoacoustic stereo bass management for the Kokoro chorus system.
//!
//! Splits stereo audio via an LR4 crossover, mono-sums bass for phase
//! coherence, optionally applies harmonic enhancement and allpass phase
//! rotation for perceived width, filters subsonic rumble, and recombines.
//!
//! References: Linkwitz (JAES 1976), Zolzer "DAFX" Ch.2/5, Waves MaxxBass.
//!
//! Part of #4582, #3351.

use crate::kokoro_error::KokoroError;
use crate::kokoro_tts::KOKORO_SAMPLE_RATE;

// -- Biquad infrastructure (local, avoids coupling to dynamics_filters) ------

#[derive(Debug, Clone, Copy)]
struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Debug, Clone)]
struct Biquad {
    c: BiquadCoeffs,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn new(c: BiquadCoeffs) -> Self {
        Self {
            c,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        let y = self.c.b0 * x + self.z1;
        self.z1 = self.c.b1 * x - self.c.a1 * y + self.z2;
        self.z2 = self.c.b2 * x - self.c.a2 * y;
        if !y.is_finite() {
            self.z1 = 0.0;
            self.z2 = 0.0;
            return 0.0;
        }
        y
    }

    fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// -- Butterworth coefficient computation (f64 precision) ---------------------

/// Shared Butterworth intermediate values.
fn bw_intermediates(freq_hz: f32, sample_rate: f32) -> (f64, f64, f64) {
    let w0 = 2.0_f64 * std::f64::consts::PI * f64::from(freq_hz) / f64::from(sample_rate);
    let alpha = w0.sin() / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    (w0.cos(), alpha, 1.0 + alpha)
}

fn butterworth_lp(freq_hz: f32, sample_rate: f32) -> BiquadCoeffs {
    let (cos_w0, alpha, a0) = bw_intermediates(freq_hz, sample_rate);
    BiquadCoeffs {
        b0: ((1.0 - cos_w0) / 2.0 / a0) as f32,
        b1: ((1.0 - cos_w0) / a0) as f32,
        b2: ((1.0 - cos_w0) / 2.0 / a0) as f32,
        a1: ((-2.0 * cos_w0) / a0) as f32,
        a2: ((1.0 - alpha) / a0) as f32,
    }
}

fn butterworth_hp(freq_hz: f32, sample_rate: f32) -> BiquadCoeffs {
    let (cos_w0, alpha, a0) = bw_intermediates(freq_hz, sample_rate);
    BiquadCoeffs {
        b0: (f64::midpoint(1.0, cos_w0) / a0) as f32,
        b1: ((-(1.0 + cos_w0)) / a0) as f32,
        b2: (f64::midpoint(1.0, cos_w0) / a0) as f32,
        a1: ((-2.0 * cos_w0) / a0) as f32,
        a2: ((1.0 - alpha) / a0) as f32,
    }
}

fn butterworth_allpass(freq_hz: f32, sample_rate: f32) -> BiquadCoeffs {
    let (cos_w0, alpha, a0) = bw_intermediates(freq_hz, sample_rate);
    BiquadCoeffs {
        b0: ((1.0 - alpha) / a0) as f32,
        b1: ((-2.0 * cos_w0) / a0) as f32,
        b2: 1.0,
        a1: ((-2.0 * cos_w0) / a0) as f32,
        a2: ((1.0 - alpha) / a0) as f32,
    }
}

// -- LR4 filter (2 cascaded Butterworth 2nd-order, -24 dB/oct) ---------------

#[derive(Debug, Clone)]
struct Lr4Filter {
    stage1: Biquad,
    stage2: Biquad,
}

impl Lr4Filter {
    fn lowpass(f: f32, sr: f32) -> Self {
        let c = butterworth_lp(f, sr);
        Self {
            stage1: Biquad::new(c),
            stage2: Biquad::new(c),
        }
    }
    fn highpass(f: f32, sr: f32) -> Self {
        let c = butterworth_hp(f, sr);
        Self {
            stage1: Biquad::new(c),
            stage2: Biquad::new(c),
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.stage2.process(self.stage1.process(x))
    }
    fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }
}

// -- 18 dB/oct rumble filter: biquad HP + one-pole HP cascade ----------------

#[derive(Debug, Clone)]
struct OnePoleHP {
    coeff: f32,
    x_prev: f32,
    y_prev: f32,
}

impl OnePoleHP {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
        Self {
            coeff: rc / (rc + 1.0 / sample_rate),
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        if !x.is_finite() {
            self.x_prev = 0.0;
            self.y_prev = 0.0;
            return 0.0;
        }
        let y = self.coeff * (self.y_prev + x - self.x_prev);
        self.x_prev = x;
        self.y_prev = if y.is_finite() { y } else { 0.0 };
        self.y_prev
    }
    fn reset(&mut self) {
        self.x_prev = 0.0;
        self.y_prev = 0.0;
    }
}

#[derive(Debug, Clone)]
struct RumbleFilter {
    biquad: Biquad,
    one_pole: OnePoleHP,
}

impl RumbleFilter {
    fn new(f: f32, sr: f32) -> Self {
        Self {
            biquad: Biquad::new(butterworth_hp(f, sr)),
            one_pole: OnePoleHP::new(f, sr),
        }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        self.one_pole.process(self.biquad.process(x))
    }
    fn reset(&mut self) {
        self.biquad.reset();
        self.one_pole.reset();
    }
}

// -- Configuration -----------------------------------------------------------

/// Configuration for psychoacoustic stereo bass management.
///
/// Constructed via [`BassManagementConfig::new`] (required for cross-crate
/// use due to `#[non_exhaustive]`).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct BassManagementConfig {
    /// Crossover frequency (Hz). Range: 60-300. Default: 120.
    pub crossover_hz: f32,
    /// Mono-sum bass below crossover. Default: true.
    pub mono_below: bool,
    /// Allpass phase rotation for perceived stereo width. Default: false.
    pub phase_trick: bool,
    /// Time-align sub with mids via allpass compensation. Default: true.
    pub sub_alignment: bool,
    /// Rumble HPF cutoff (Hz), 18 dB/oct. Range: 10-60. Default: 30.
    pub rumble_filter_hz: f32,
    /// Even-harmonic bass enhancement (0-1). Default: 0.
    pub bass_enhancement: f32,
}

impl Default for BassManagementConfig {
    fn default() -> Self {
        Self {
            crossover_hz: 120.0,
            mono_below: true,
            phase_trick: false,
            sub_alignment: true,
            rumble_filter_hz: 30.0,
            bass_enhancement: 0.0,
        }
    }
}

impl BassManagementConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the crossover frequency in Hz.
    #[must_use]
    pub fn with_crossover_hz(mut self, hz: f32) -> Self {
        self.crossover_hz = hz;
        self
    }
    /// Enable or disable mono-summing below the crossover.
    #[must_use]
    pub fn with_mono_below(mut self, mono: bool) -> Self {
        self.mono_below = mono;
        self
    }
    /// Enable or disable phase-rotation trick for perceived stereo width.
    #[must_use]
    pub fn with_phase_trick(mut self, v: bool) -> Self {
        self.phase_trick = v;
        self
    }
    /// Enable or disable sub-alignment (allpass delay compensation).
    #[must_use]
    pub fn with_sub_alignment(mut self, v: bool) -> Self {
        self.sub_alignment = v;
        self
    }
    /// Set the rumble filter cutoff in Hz.
    #[must_use]
    pub fn with_rumble_filter_hz(mut self, hz: f32) -> Self {
        self.rumble_filter_hz = hz;
        self
    }
    /// Set the bass enhancement amount (0.0-1.0).
    #[must_use]
    pub fn with_bass_enhancement(mut self, v: f32) -> Self {
        self.bass_enhancement = v;
        self
    }

    /// Validate all parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let err = |field, reason: String| KokoroError::InvalidConfig { field, reason };
        if !self.crossover_hz.is_finite() || self.crossover_hz < 60.0 || self.crossover_hz > 300.0 {
            return Err(err(
                "crossover_hz",
                format!("{}: must be in [60, 300]", self.crossover_hz),
            ));
        }
        if !self.rumble_filter_hz.is_finite()
            || self.rumble_filter_hz < 10.0
            || self.rumble_filter_hz > 60.0
        {
            return Err(err(
                "rumble_filter_hz",
                format!("{}: must be in [10, 60]", self.rumble_filter_hz),
            ));
        }
        if !self.bass_enhancement.is_finite()
            || self.bass_enhancement < 0.0
            || self.bass_enhancement > 1.0
        {
            return Err(err(
                "bass_enhancement",
                format!("{}: must be in [0, 1]", self.bass_enhancement),
            ));
        }
        if self.rumble_filter_hz >= self.crossover_hz {
            return Err(err(
                "rumble_filter_hz",
                format!(
                    "{} must be < crossover_hz {}",
                    self.rumble_filter_hz, self.crossover_hz
                ),
            ));
        }
        Ok(())
    }

    // -- Presets ----------------------------------------------------------------

    /// Broadcast: strict mono bass, 80 Hz crossover, no enhancement.
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            crossover_hz: 80.0,
            rumble_filter_hz: 40.0,
            ..Self::default()
        }
    }
    /// Headphones: phase trick for wider bass, 100 Hz crossover.
    #[must_use]
    pub fn headphones() -> Self {
        Self {
            crossover_hz: 100.0,
            phase_trick: true,
            rumble_filter_hz: 25.0,
            bass_enhancement: 0.15,
            ..Self::default()
        }
    }
    /// Small speakers: higher crossover, harmonic enhancement for perceived bass.
    #[must_use]
    pub fn speakers_small() -> Self {
        Self {
            crossover_hz: 150.0,
            rumble_filter_hz: 35.0,
            bass_enhancement: 0.3,
            ..Self::default()
        }
    }
    /// Large speakers: standard 120 Hz crossover, no enhancement.
    #[must_use]
    pub fn speakers_large() -> Self {
        Self {
            rumble_filter_hz: 25.0,
            ..Self::default()
        }
    }
    /// Subwoofer mix: 80 Hz crossover, phase trick, very low rumble filter.
    #[must_use]
    pub fn subwoofer_mix() -> Self {
        Self {
            crossover_hz: 80.0,
            phase_trick: true,
            rumble_filter_hz: 20.0,
            ..Self::default()
        }
    }
}

// -- BassManager processor ---------------------------------------------------

/// Stateful psychoacoustic stereo bass manager.
///
/// Splits stereo input via LR4 crossover, mono-sums bass, optionally applies
/// harmonic enhancement and phase rotation, filters rumble, and recombines.
#[derive(Debug, Clone)]
pub struct BassManager {
    config: BassManagementConfig,
    lp_left: Lr4Filter,
    hp_left: Lr4Filter,
    lp_right: Lr4Filter,
    hp_right: Lr4Filter,
    rumble: RumbleFilter,
    phase_allpass: Biquad,
    align_ap_left: [Biquad; 2],
    align_ap_right: [Biquad; 2],
}

impl BassManager {
    /// Create a new bass manager.
    ///
    /// # Errors
    /// Returns `KokoroError::InvalidConfig` if parameters are out of range.
    pub fn new(config: BassManagementConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("{sample_rate}: must be finite and positive"),
            });
        }
        let (sr, xover) = (sample_rate, config.crossover_hz);
        let ap = butterworth_allpass(xover, sr);
        Ok(Self {
            config,
            lp_left: Lr4Filter::lowpass(xover, sr),
            hp_left: Lr4Filter::highpass(xover, sr),
            lp_right: Lr4Filter::lowpass(xover, sr),
            hp_right: Lr4Filter::highpass(xover, sr),
            rumble: RumbleFilter::new(config.rumble_filter_hz, sr),
            phase_allpass: Biquad::new(ap),
            align_ap_left: [Biquad::new(ap), Biquad::new(ap)],
            align_ap_right: [Biquad::new(ap), Biquad::new(ap)],
        })
    }

    /// Create using Kokoro's default 24 kHz sample rate.
    pub fn new_kokoro(config: BassManagementConfig) -> Result<Self, KokoroError> {
        Self::new(config, KOKORO_SAMPLE_RATE as f32)
    }

    /// Process stereo audio in-place.
    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        let len = left.len().min(right.len());
        for i in 0..len {
            let (l_in, r_in) = (left[i], right[i]);

            // LR4 crossover split
            let bass_l = self.lp_left.process(l_in);
            let bass_r = self.lp_right.process(r_in);
            let mut mid_high_l = self.hp_left.process(l_in);
            let mut mid_high_r = self.hp_right.process(r_in);

            // Sub-alignment: compensate LR4 group delay on mid+high
            if self.config.sub_alignment {
                for ap in &mut self.align_ap_left {
                    mid_high_l = ap.process(mid_high_l);
                }
                for ap in &mut self.align_ap_right {
                    mid_high_r = ap.process(mid_high_r);
                }
            }

            // Mono-sum bass or pass through stereo
            let (bass_out_l, bass_out_r) = if self.config.mono_below {
                let m = (bass_l + bass_r) * 0.5;
                (m, m)
            } else {
                (bass_l, bass_r)
            };

            // Harmonic enhancement (even-harmonic saturation)
            let enh_l = harmonic_enhance(bass_out_l, self.config.bass_enhancement);
            let enh_r = harmonic_enhance(bass_out_r, self.config.bass_enhancement);

            // Rumble filter
            let rum_l = self.rumble.process(enh_l);
            let rum_r = if self.config.mono_below {
                rum_l
            } else {
                self.rumble.process(enh_r)
            };

            // Phase rotation for perceived stereo width
            let (fin_l, fin_r) = if self.config.phase_trick {
                (rum_l, self.phase_allpass.process(rum_l))
            } else {
                (rum_l, rum_r)
            };

            // Recombine with NaN/Inf guard
            left[i] = mid_high_l + fin_l;
            right[i] = mid_high_r + fin_r;
            if !left[i].is_finite() {
                left[i] = 0.0;
            }
            if !right[i].is_finite() {
                right[i] = 0.0;
            }
        }
    }

    /// Reset all internal filter state.
    pub fn reset(&mut self) {
        self.lp_left.reset();
        self.hp_left.reset();
        self.lp_right.reset();
        self.hp_right.reset();
        self.rumble.reset();
        self.phase_allpass.reset();
        for ap in &mut self.align_ap_left {
            ap.reset();
        }
        for ap in &mut self.align_ap_right {
            ap.reset();
        }
    }

    /// Read-only access to the current configuration.
    #[must_use]
    pub fn config(&self) -> &BassManagementConfig {
        &self.config
    }
}

// -- Harmonic enhancement (even-harmonic saturation) -------------------------

/// Gentle even-harmonic saturation for bass warmth. Uses soft-clip curve
/// `x / (1 + |x|)` producing 2nd-harmonic content that the ear interprets
/// as bass presence even on speakers that cannot reproduce the fundamental.
#[inline]
fn harmonic_enhance(x: f32, amount: f32) -> f32 {
    if amount < 1e-6 || !x.is_finite() {
        return x;
    }
    let saturated = x / (1.0 + x.abs());
    x * (1.0 - amount) + saturated * amount
}

#[cfg(test)]
#[path = "kokoro_chorus_bass_management_tests.rs"]
mod tests;

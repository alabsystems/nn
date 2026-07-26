// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Linkwitz-Riley crossover filter infrastructure for multi-band dynamics.
//!
//! Provides Biquad filters, Butterworth coefficient computation, LR4
//! (Linkwitz-Riley 4th-order) lowpass/highpass filters, allpass phase
//! compensation, and a 3-band crossover splitter.
//!
//! All filter coefficient calculations use f64 to avoid precision loss,
//! then cast to f32 for processing. All sample-processing functions
//! check `is_finite()` for IEEE 754 safety.

// ---------------------------------------------------------------------------
// Biquad filter (local copy -- kept private to avoid coupling to EQ module)
// ---------------------------------------------------------------------------

/// Second-order IIR biquad coefficients (normalized: a0 = 1).
#[derive(Debug, Clone, Copy)]
pub(super) struct BiquadCoeffs {
    pub(super) b0: f32,
    pub(super) b1: f32,
    pub(super) b2: f32,
    pub(super) a1: f32,
    pub(super) a2: f32,
}

/// Biquad filter state (Direct Form II Transposed).
#[derive(Debug, Clone)]
pub(super) struct Biquad {
    c: BiquadCoeffs,
    z1: f32,
    z2: f32,
}

impl Biquad {
    pub(super) fn new(c: BiquadCoeffs) -> Self {
        Self {
            c,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    pub(super) fn process(&mut self, x: f32) -> f32 {
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

    pub(super) fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Second-order allpass filter
// ---------------------------------------------------------------------------

/// Compute 2nd-order allpass coefficients matching a Butterworth filter at
/// the given frequency. The allpass has unity magnitude at all frequencies
/// but matches the phase response of the Butterworth LP (or HP) filter.
///
/// Transfer function: H(z) = (a2 + a1*z^-1 + z^-2) / (1 + a1*z^-1 + a2*z^-2)
fn butterworth_allpass_coeffs(freq_hz: f32, sample_rate: f32) -> BiquadCoeffs {
    let w0 = 2.0_f64 * std::f64::consts::PI * f64::from(freq_hz) / f64::from(sample_rate);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    // Allpass: numerator = reverse of denominator (normalized by a0).
    BiquadCoeffs {
        b0: (a2 / a0) as f32,
        b1: (a1 / a0) as f32,
        b2: 1.0, // a0/a0
        a1: (a1 / a0) as f32,
        a2: (a2 / a0) as f32,
    }
}

/// 4th-order allpass: two cascaded 2nd-order allpass filters.
///
/// Matches the group delay of an LR4 (Linkwitz-Riley 4th-order) crossover
/// stage at the given frequency. Used for phase compensation in 3-band
/// crossover designs.
#[derive(Debug, Clone)]
struct Lr4Allpass {
    stage1: Biquad,
    stage2: Biquad,
}

impl Lr4Allpass {
    fn new(freq_hz: f32, sample_rate: f32) -> Self {
        let c = butterworth_allpass_coeffs(freq_hz, sample_rate);
        Self {
            stage1: Biquad::new(c),
            stage2: Biquad::new(c),
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.stage1.process(x);
        self.stage2.process(y)
    }

    fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }
}

// ---------------------------------------------------------------------------
// Butterworth lowpass / highpass coefficient computation (f64 precision)
// ---------------------------------------------------------------------------

/// Compute 2nd-order Butterworth lowpass coefficients.
///
/// Uses f64 for intermediate computation to avoid precision loss at low
/// frequencies where `cos(w0)` is close to 1.
fn butterworth_lp(freq_hz: f32, sample_rate: f32) -> BiquadCoeffs {
    let w0 = 2.0_f64 * std::f64::consts::PI * f64::from(freq_hz) / f64::from(sample_rate);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    // Butterworth Q = 1/sqrt(2)
    let alpha = sin_w0 / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: ((1.0 - cos_w0) / 2.0 / a0) as f32,
        b1: ((1.0 - cos_w0) / a0) as f32,
        b2: ((1.0 - cos_w0) / 2.0 / a0) as f32,
        a1: ((-2.0 * cos_w0) / a0) as f32,
        a2: ((1.0 - alpha) / a0) as f32,
    }
}

/// Compute 2nd-order Butterworth highpass coefficients.
fn butterworth_hp(freq_hz: f32, sample_rate: f32) -> BiquadCoeffs {
    let w0 = 2.0_f64 * std::f64::consts::PI * f64::from(freq_hz) / f64::from(sample_rate);
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * std::f64::consts::FRAC_1_SQRT_2);
    let a0 = 1.0 + alpha;
    BiquadCoeffs {
        b0: (f64::midpoint(1.0, cos_w0) / a0) as f32,
        b1: ((-(1.0 + cos_w0)) / a0) as f32,
        b2: (f64::midpoint(1.0, cos_w0) / a0) as f32,
        a1: ((-2.0 * cos_w0) / a0) as f32,
        a2: ((1.0 - alpha) / a0) as f32,
    }
}

// ---------------------------------------------------------------------------
// Linkwitz-Riley 4th-order crossover (two cascaded Butterworth)
// ---------------------------------------------------------------------------

/// A single Linkwitz-Riley 4th-order filter (low or high).
///
/// Two cascaded 2nd-order Butterworth filters produce a 4th-order slope
/// (-24 dB/oct) with -6 dB at the crossover frequency.
#[derive(Debug, Clone)]
struct Lr4Filter {
    stage1: Biquad,
    stage2: Biquad,
}

impl Lr4Filter {
    fn lowpass(freq_hz: f32, sample_rate: f32) -> Self {
        let c = butterworth_lp(freq_hz, sample_rate);
        Self {
            stage1: Biquad::new(c),
            stage2: Biquad::new(c),
        }
    }

    fn highpass(freq_hz: f32, sample_rate: f32) -> Self {
        let c = butterworth_hp(freq_hz, sample_rate);
        Self {
            stage1: Biquad::new(c),
            stage2: Biquad::new(c),
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.stage1.process(x);
        self.stage2.process(y)
    }

    fn reset(&mut self) {
        self.stage1.reset();
        self.stage2.reset();
    }
}

/// 3-band Linkwitz-Riley crossover with allpass phase compensation.
///
/// Split point 1: `low_freq` (low vs mid+high).
/// Split point 2: `high_freq` (mid vs high), applied to the mid+high band.
///
/// The low band is phase-compensated via a 4th-order allpass matching the
/// group delay of the second crossover stage. This ensures that
/// `low + mid + high` reconstructs the input with correct phase alignment.
#[derive(Debug, Clone)]
pub(super) struct ThreeBandCrossover {
    /// LR4 lowpass at low_freq.
    lp1: Lr4Filter,
    /// LR4 highpass at low_freq.
    hp1: Lr4Filter,
    /// LR4 lowpass at high_freq (applied to hp1 output for mid).
    lp2: Lr4Filter,
    /// LR4 highpass at high_freq (applied to hp1 output for high).
    hp2: Lr4Filter,
    /// 4th-order allpass at high_freq for low-band phase compensation.
    ap: Lr4Allpass,
}

impl ThreeBandCrossover {
    pub(super) fn new(low_freq: f32, high_freq: f32, sample_rate: f32) -> Self {
        Self {
            lp1: Lr4Filter::lowpass(low_freq, sample_rate),
            hp1: Lr4Filter::highpass(low_freq, sample_rate),
            lp2: Lr4Filter::lowpass(high_freq, sample_rate),
            hp2: Lr4Filter::highpass(high_freq, sample_rate),
            ap: Lr4Allpass::new(high_freq, sample_rate),
        }
    }

    /// Split a single sample into (low, mid, high).
    #[inline]
    pub(super) fn split(&mut self, x: f32) -> (f32, f32, f32) {
        let low_raw = self.lp1.process(x);
        let mid_high = self.hp1.process(x);
        let mid = self.lp2.process(mid_high);
        let high = self.hp2.process(mid_high);
        // Phase-compensate low band to match the delay of the 2nd crossover.
        let low = self.ap.process(low_raw);
        (low, mid, high)
    }

    pub(super) fn reset(&mut self) {
        self.lp1.reset();
        self.hp1.reset();
        self.lp2.reset();
        self.hp2.reset();
        self.ap.reset();
    }
}

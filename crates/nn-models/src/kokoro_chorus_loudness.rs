// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Psychoacoustic loudness model with ITU-R BS.1770 LUFS measurement.
//!
//! Implements K-weighting (ITU-R BS.1770-4), A-weighting (IEC 61672),
//! true peak detection (4x oversampled), and Bark-scale critical band
//! analysis for the Kokoro chorus dynamics pipeline.
//!
//! # References
//!
//! - ITU-R BS.1770-4, "Algorithms to measure audio programme loudness
//!   and true-peak audio level," ITU, 2015.
//! - IEC 61672-1:2013, "Electroacoustics — Sound level meters."
//! - Zwicker & Fastl, "Psychoacoustics," 3rd ed., Springer, 2007.

use crate::kokoro_error::KokoroError;

const SILENCE_DB: f32 = -120.0;
const AMPLITUDE_FLOOR: f64 = 1e-20;
const LUFS_OFFSET: f64 = -0.691;

fn block_size(sample_rate: f32) -> usize {
    (sample_rate * 0.4).round() as usize
}

/// Bark-scale critical band edges (Hz), 25 values for 24 bands.
/// Source: Zwicker & Fastl, Table 6.1.
const BARK_EDGES: [f32; 25] = [
    0.0, 100.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0,
    2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0,
    15500.0,
];

// ---------------------------------------------------------------------------
// Biquad filter (Direct Form II Transposed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BiquadFilter {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl BiquadFilter {
    fn new(b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
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
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
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

    fn process_buffer(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&s| self.process(s)).collect()
    }
}

// ---------------------------------------------------------------------------
// K-weighting (ITU-R BS.1770-4): pre-filter + RLB high-pass
// ---------------------------------------------------------------------------

/// Stage 1: high-shelf +3.999 dB above ~1681 Hz (head acoustic effect).
fn k_weight_pre_filter(sr: f32) -> BiquadFilter {
    let sr = f64::from(sr);
    let (db, f0, q) = (3.999843853973347, 1681.974450955533, 0.7071752369554196);
    let k = (std::f64::consts::PI * f0 / sr).tan();
    let vh = 10.0f64.powf(db / 20.0);
    let vb = vh.powf(0.4996667741545416);
    let (k2, a0i) = (k * k, 1.0 / (1.0 + k / q + k * k));
    BiquadFilter::new(
        ((vh + vb * k / q + k2) * a0i) as f32,
        (2.0 * (k2 - vh) * a0i) as f32,
        ((vh - vb * k / q + k2) * a0i) as f32,
        (2.0 * (k2 - 1.0) * a0i) as f32,
        ((1.0 - k / q + k2) * a0i) as f32,
    )
}

/// Stage 2: RLB high-pass at ~38 Hz (removes sub-bass).
fn k_weight_rlb_filter(sr: f32) -> BiquadFilter {
    let sr = f64::from(sr);
    let (f0, q) = (38.13547087602444, 0.5003270373238773);
    let k = (std::f64::consts::PI * f0 / sr).tan();
    let (k2, a0i) = (k * k, 1.0 / (1.0 + k / q + k * k));
    BiquadFilter::new(
        a0i as f32,
        (-2.0 * a0i) as f32,
        a0i as f32,
        (2.0 * (k2 - 1.0) * a0i) as f32,
        ((1.0 - k / q + k2) * a0i) as f32,
    )
}

// ---------------------------------------------------------------------------
// A-weighting (IEC 61672): 4 cascaded biquad sections
// ---------------------------------------------------------------------------

fn a_weight_filters(sr: f32) -> Vec<BiquadFilter> {
    let sr = f64::from(sr);
    let pi = std::f64::consts::PI;
    let sqrt2 = std::f64::consts::SQRT_2;

    let hp = |f0: f64| -> BiquadFilter {
        let k = (pi * f0 / sr).tan();
        let (k2, a0i) = (k * k, 1.0 / (1.0 + sqrt2 * k + k * k));
        BiquadFilter::new(
            a0i as f32,
            (-2.0 * a0i) as f32,
            a0i as f32,
            (2.0 * (k2 - 1.0) * a0i) as f32,
            ((1.0 - sqrt2 * k + k2) * a0i) as f32,
        )
    };
    let lp = |f0: f64| -> BiquadFilter {
        let k = (pi * f0 / sr).tan();
        let (k2, a0i) = (k * k, 1.0 / (1.0 + sqrt2 * k + k * k));
        BiquadFilter::new(
            (k2 * a0i) as f32,
            (2.0 * k2 * a0i) as f32,
            (k2 * a0i) as f32,
            (2.0 * (k2 - 1.0) * a0i) as f32,
            ((1.0 - sqrt2 * k + k2) * a0i) as f32,
        )
    };

    vec![hp(20.598997), hp(107.65265), lp(12194.217), lp(12194.217)]
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Frequency weighting mode for loudness measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum LoudnessWeighting {
    /// ITU-R BS.1770 K-weighting (broadcast standard).
    #[default]
    KWeighting,
    /// A-weighting per IEC 61672 (human hearing sensitivity).
    AWeighting,
    /// Flat (no weighting, raw RMS).
    Flat,
}


/// Configuration for psychoacoustic loudness measurement.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoudnessConfig {
    /// Target integrated loudness in LUFS (default -16.0).
    pub target_lufs: f32,
    /// True peak ceiling in dBFS (default -1.0).
    pub true_peak_limit_dbfs: f32,
    /// Absolute gate threshold in LUFS (default -70.0).
    pub gate_threshold_lufs: f32,
    /// Momentary measurement window in ms (default 400.0).
    pub measurement_window_ms: f32,
    /// Frequency weighting mode (default K-weighting).
    pub weighting: LoudnessWeighting,
}

impl Default for LoudnessConfig {
    fn default() -> Self {
        Self {
            target_lufs: -16.0,
            true_peak_limit_dbfs: -1.0,
            gate_threshold_lufs: -70.0,
            measurement_window_ms: 400.0,
            weighting: LoudnessWeighting::KWeighting,
        }
    }
}

impl LoudnessConfig {
    /// Create a validated loudness config.
    pub fn new(
        target_lufs: f32,
        true_peak_limit_dbfs: f32,
        gate_threshold_lufs: f32,
        measurement_window_ms: f32,
        weighting: LoudnessWeighting,
    ) -> Result<Self, KokoroError> {
        let cfg = Self {
            target_lufs,
            true_peak_limit_dbfs,
            gate_threshold_lufs,
            measurement_window_ms,
            weighting,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    #[must_use]
    pub fn with_target_lufs(mut self, v: f32) -> Self {
        self.target_lufs = v;
        self
    }
    #[must_use]
    pub fn with_true_peak_limit(mut self, v: f32) -> Self {
        self.true_peak_limit_dbfs = v;
        self
    }
    #[must_use]
    pub fn with_gate_threshold(mut self, v: f32) -> Self {
        self.gate_threshold_lufs = v;
        self
    }
    #[must_use]
    pub fn with_measurement_window_ms(mut self, v: f32) -> Self {
        self.measurement_window_ms = v;
        self
    }
    #[must_use]
    pub fn with_weighting(mut self, w: LoudnessWeighting) -> Self {
        self.weighting = w;
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), KokoroError> {
        let check = |field: &'static str, v: f32, lo: f32, hi: f32| -> Result<(), KokoroError> {
            if !v.is_finite() || v < lo || v > hi {
                return Err(KokoroError::InvalidConfig {
                    field,
                    reason: format!("must be finite in [{lo}, {hi}], got {v}"),
                });
            }
            Ok(())
        };
        check("target_lufs", self.target_lufs, -60.0, 0.0)?;
        check(
            "true_peak_limit_dbfs",
            self.true_peak_limit_dbfs,
            -12.0,
            0.0,
        )?;
        check(
            "gate_threshold_lufs",
            self.gate_threshold_lufs,
            -120.0,
            -10.0,
        )?;
        check(
            "measurement_window_ms",
            self.measurement_window_ms,
            50.0,
            3000.0,
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LoudnessMeter
// ---------------------------------------------------------------------------

/// ITU-R BS.1770-4 loudness meter with K/A/flat weighting.
pub struct LoudnessMeter {
    config: LoudnessConfig,
    sample_rate: f32,
    pre_filter: BiquadFilter,
    rlb_filter: BiquadFilter,
    a_weight_filters: Vec<BiquadFilter>,
    block_powers: Vec<f32>,
    sample_count: usize,
    current_block_sum: f64,
    current_block_count: usize,
}

impl LoudnessMeter {
    /// Create a meter for the given sample rate.
    pub fn new(config: &LoudnessConfig, sample_rate: f32) -> Result<Self, KokoroError> {
        config.validate()?;
        if !sample_rate.is_finite() || sample_rate <= 0.0 {
            return Err(KokoroError::InvalidConfig {
                field: "sample_rate",
                reason: format!("must be finite and positive, got {sample_rate}"),
            });
        }
        Ok(Self {
            config: config.clone(),
            sample_rate,
            pre_filter: k_weight_pre_filter(sample_rate),
            rlb_filter: k_weight_rlb_filter(sample_rate),
            a_weight_filters: a_weight_filters(sample_rate),
            block_powers: Vec::new(),
            sample_count: 0,
            current_block_sum: 0.0,
            current_block_count: 0,
        })
    }

    /// Create a meter with default config at 24 kHz (Kokoro sample rate).
    pub fn default_24k() -> Result<Self, KokoroError> {
        Self::new(&LoudnessConfig::default(), 24000.0)
    }

    /// Reset all measurement state (filters + block history).
    pub fn reset(&mut self) {
        self.pre_filter.reset();
        self.rlb_filter.reset();
        for f in &mut self.a_weight_filters {
            f.reset();
        }
        self.block_powers.clear();
        self.sample_count = 0;
        self.current_block_sum = 0.0;
        self.current_block_count = 0;
    }

    fn apply_weighting(&mut self, audio: &[f32]) -> Vec<f32> {
        match self.config.weighting {
            LoudnessWeighting::KWeighting => {
                let s1 = self.pre_filter.process_buffer(audio);
                self.rlb_filter.process_buffer(&s1)
            }
            LoudnessWeighting::AWeighting => {
                let mut buf = audio.to_vec();
                for f in &mut self.a_weight_filters {
                    buf = f.process_buffer(&buf);
                }
                buf
            }
            LoudnessWeighting::Flat => audio.to_vec(),
        }
    }

    /// Measure momentary loudness (LUFS) over a chunk. Does not update integrated state.
    #[must_use]
    pub fn measure_momentary(&mut self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return SILENCE_DB;
        }
        let w = self.apply_weighting(audio);
        let (sum, n) = w.iter().fold((0.0f64, 0u64), |(a, n), &s| {
            if s.is_finite() {
                (a + f64::from(s) * f64::from(s), n + 1)
            } else {
                (a, n)
            }
        });
        if n == 0 || sum < AMPLITUDE_FLOOR {
            return SILENCE_DB;
        }
        let lufs = LUFS_OFFSET + 10.0 * (sum / n as f64).log10();
        if lufs.is_finite() {
            lufs as f32
        } else {
            SILENCE_DB
        }
    }

    /// Feed audio for integrated loudness (accumulates gated 400ms blocks).
    pub fn feed(&mut self, audio: &[f32]) {
        let weighted = self.apply_weighting(audio);
        let bs = block_size(self.sample_rate);
        if bs == 0 {
            return;
        }
        for &s in &weighted {
            if !s.is_finite() {
                continue;
            }
            self.current_block_sum += f64::from(s) * f64::from(s);
            self.current_block_count += 1;
            self.sample_count += 1;
            if self.current_block_count >= bs {
                let mean_sq = self.current_block_sum / self.current_block_count as f64;
                let bl = LUFS_OFFSET + 10.0 * mean_sq.max(AMPLITUDE_FLOOR).log10();
                if bl.is_finite() && bl >= f64::from(self.config.gate_threshold_lufs) {
                    self.block_powers.push(mean_sq as f32);
                }
                self.current_block_sum = 0.0;
                self.current_block_count = 0;
            }
        }
    }

    /// Integrated loudness with BS.1770 two-stage gating.
    #[must_use]
    pub fn integrated_loudness(&self) -> f32 {
        if self.block_powers.is_empty() {
            return SILENCE_DB;
        }
        let ungated_mean: f64 = self.block_powers.iter().map(|&p| f64::from(p)).sum::<f64>()
            / self.block_powers.len() as f64;
        let ungated_lufs = LUFS_OFFSET + 10.0 * ungated_mean.max(AMPLITUDE_FLOOR).log10();
        let rel_gate = ungated_lufs - 10.0;
        let (gs, gc) = self.block_powers.iter().fold((0.0f64, 0u64), |(a, n), &p| {
            let bl = LUFS_OFFSET + 10.0 * f64::from(p).max(AMPLITUDE_FLOOR).log10();
            if bl >= rel_gate {
                (a + f64::from(p), n + 1)
            } else {
                (a, n)
            }
        });
        if gc == 0 {
            return SILENCE_DB;
        }
        let lufs = LUFS_OFFSET + 10.0 * (gs / gc as f64).max(AMPLITUDE_FLOOR).log10();
        if lufs.is_finite() {
            lufs as f32
        } else {
            SILENCE_DB
        }
    }

    /// Feed audio and return current integrated loudness.
    #[must_use]
    pub fn measure_integrated(&mut self, audio: &[f32]) -> f32 {
        self.feed(audio);
        self.integrated_loudness()
    }

    /// Normalize audio to target LUFS, respecting true peak ceiling. Returns applied gain dB.
    pub fn normalize_to_target(&mut self, audio: &mut [f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }
        self.reset();
        self.feed(audio);
        let current = self.integrated_loudness();
        if current <= SILENCE_DB + 1.0 {
            return 0.0;
        }
        let needed = self.config.target_lufs - current;
        let tp = measure_true_peak(audio, self.sample_rate);
        let headroom = self.config.true_peak_limit_dbfs - tp;
        let gain_db = needed.min(headroom);
        if !gain_db.is_finite() || gain_db.abs() < 0.01 {
            return 0.0;
        }
        let gain_lin = 10.0f32.powf(gain_db / 20.0);
        for s in audio.iter_mut() {
            if s.is_finite() {
                *s *= gain_lin;
            } else {
                *s = 0.0;
            }
        }
        gain_db
    }

    #[must_use]
    pub fn config(&self) -> &LoudnessConfig {
        &self.config
    }
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// True peak (ITU-R BS.1770 Annex 2, 4x oversampled)
// ---------------------------------------------------------------------------

/// Measure true peak via 4x cubic Hermite oversampling. Returns dBFS.
#[must_use]
pub fn measure_true_peak(audio: &[f32], _sample_rate: f32) -> f32 {
    if audio.is_empty() {
        return SILENCE_DB;
    }
    let mut max_abs: f64 = 0.0;
    for i in 0..audio.len() {
        let s = audio[i];
        if !s.is_finite() {
            continue;
        }
        let a = f64::from(s).abs();
        if a > max_abs {
            max_abs = a;
        }
        if i + 1 < audio.len() && audio[i + 1].is_finite() {
            let y0 = if i > 0 && audio[i - 1].is_finite() {
                f64::from(audio[i - 1])
            } else {
                f64::from(s)
            };
            let (y1, y2) = (f64::from(s), f64::from(audio[i + 1]));
            let y3 = if i + 2 < audio.len() && audio[i + 2].is_finite() {
                f64::from(audio[i + 2])
            } else {
                y2
            };
            for k in 1..4u32 {
                let t = f64::from(k) / 4.0;
                let (c0, c1) = (y1, 0.5 * (y2 - y0));
                let c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
                let c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
                let v = ((c3 * t + c2) * t + c1) * t + c0;
                if v.abs() > max_abs {
                    max_abs = v.abs();
                }
            }
        }
    }
    if max_abs < AMPLITUDE_FLOOR {
        return SILENCE_DB;
    }
    let db = 20.0 * max_abs.log10();
    if db.is_finite() {
        db as f32
    } else {
        SILENCE_DB
    }
}

// ---------------------------------------------------------------------------
// Bark-scale critical band analysis (24 bands)
// ---------------------------------------------------------------------------

/// Per-band loudness via DFT bin mapping to 24 Bark critical bands.
/// Returns 24 power values in dB. Useful for psychoacoustic dynamics.
#[must_use]
pub fn bark_band_loudness(audio: &[f32], sample_rate: f32) -> Vec<f32> {
    if audio.is_empty() || !sample_rate.is_finite() || sample_rate <= 0.0 {
        return vec![SILENCE_DB; 24];
    }
    let n = audio.len().max(256).next_power_of_two().min(8192);
    let half = n / 2 + 1;
    let bin_hz = f64::from(sample_rate) / n as f64;

    // Power spectrum via DFT with Hann window.
    let mut pspec = vec![0.0f64; half];
    for k in 0..half {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let fk = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        for (i, &s) in audio.iter().enumerate() {
            if !s.is_finite() {
                continue;
            }
            let w = if n > 1 {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
            } else {
                1.0
            };
            let x = f64::from(s) * w;
            let ang = fk * i as f64;
            re += x * ang.cos();
            im += x * ang.sin();
        }
        pspec[k] = (re * re + im * im) / n as f64;
    }

    let mut result = vec![SILENCE_DB; 24];
    for band in 0..24 {
        let (lo, hi) = (BARK_EDGES[band], BARK_EDGES[band + 1]);
        let blo = (f64::from(lo) / bin_hz).floor() as usize;
        let bhi = ((f64::from(hi) / bin_hz).ceil() as usize).min(half);
        if blo >= bhi || blo >= half {
            continue;
        }
        let bp: f64 = pspec[blo..bhi].iter().sum();
        let mp = bp / (bhi - blo) as f64;
        if mp > AMPLITUDE_FLOOR {
            let db = 10.0 * mp.log10();
            result[band] = if db.is_finite() {
                db as f32
            } else {
                SILENCE_DB
            };
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests (extracted to separate file for 500-line limit)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "kokoro_chorus_loudness_tests.rs"]
mod tests;

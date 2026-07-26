// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Statistical testing utilities for fairness measurement.
//!
//! Provides Welch's t-test, Cohen's d effect size, and Holm-Bonferroni
//! correction for multiple comparisons. No external statistics crate needed.
//!
//! References:
//! - Welch (1947) "The Generalization of Student's Problem When Several
//!   Different Population Variances Are Involved." Biometrika.
//! - Holm (1979) "A Simple Sequentially Rejective Multiple Test Procedure."
//!   Scandinavian Journal of Statistics.

use crate::error::{DspErrorKind, TtsVerifyError};

/// Welch's t-test for two independent samples with unequal variance.
///
/// Returns `(t_statistic, degrees_of_freedom, two_sided_p_value)`.
/// Uses Welch-Satterthwaite degrees of freedom approximation.
///
/// Both samples must have at least 2 elements and finite values.
pub fn welch_t_test(sample_a: &[f64], sample_b: &[f64]) -> Result<(f64, f64, f64), TtsVerifyError> {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "Welch's t-test",
            needed: 2,
            got: sample_a.len().min(sample_b.len()),
        }));
    }

    // Defense-in-depth: validate input finiteness.
    let non_finite_count = sample_a.iter().filter(|x| !x.is_finite()).count()
        + sample_b.iter().filter(|x| !x.is_finite()).count();
    if non_finite_count > 0 {
        return Err(TtsVerifyError::NonFiniteInput {
            count: non_finite_count,
        });
    }

    let n_a = sample_a.len() as f64;
    let n_b = sample_b.len() as f64;
    let mean_a = sample_a.iter().sum::<f64>() / n_a;
    let mean_b = sample_b.iter().sum::<f64>() / n_b;

    let var_a = sample_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = sample_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    let se_a = var_a / n_a;
    let se_b = var_b / n_b;
    let se_sum = se_a + se_b;

    if se_sum < f64::EPSILON {
        // Both groups have zero variance — means are exactly equal or single-value.
        return Ok((0.0, n_a + n_b - 2.0, 1.0));
    }

    let t = (mean_a - mean_b) / se_sum.sqrt();

    // Welch-Satterthwaite degrees of freedom
    let df_numer = se_sum.powi(2);
    let df_denom = se_a.powi(2) / (n_a - 1.0) + se_b.powi(2) / (n_b - 1.0);
    let df = if df_denom < f64::EPSILON {
        n_a + n_b - 2.0
    } else {
        df_numer / df_denom
    };

    let p_value = 2.0 * (1.0 - student_t_cdf(t.abs(), df));

    if !t.is_finite() || !p_value.is_finite() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "non-finite result in Welch's t-test",
        }));
    }

    Ok((t, df, p_value))
}

/// Cohen's d effect size for two independent samples.
///
/// Uses pooled standard deviation. Returns 0.0 if both samples have zero variance.
///
/// # Errors
///
/// Returns [`TtsVerifyError::Dsp`] if either sample has fewer than 2 elements.
/// Returns [`TtsVerifyError::NonFiniteInput`] if any input value is NaN/Inf.
pub fn cohens_d(sample_a: &[f64], sample_b: &[f64]) -> Result<f64, TtsVerifyError> {
    if sample_a.len() < 2 || sample_b.len() < 2 {
        return Err(TtsVerifyError::Dsp(DspErrorKind::InsufficientSamples {
            operation: "Cohen's d",
            needed: 2,
            got: sample_a.len().min(sample_b.len()),
        }));
    }

    let non_finite_count = sample_a.iter().filter(|x| !x.is_finite()).count()
        + sample_b.iter().filter(|x| !x.is_finite()).count();
    if non_finite_count > 0 {
        return Err(TtsVerifyError::NonFiniteInput {
            count: non_finite_count,
        });
    }

    let n_a = sample_a.len() as f64;
    let n_b = sample_b.len() as f64;
    let mean_a = sample_a.iter().sum::<f64>() / n_a;
    let mean_b = sample_b.iter().sum::<f64>() / n_b;

    let var_a = sample_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = sample_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    // Pooled standard deviation
    let s_pooled = (((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0)).sqrt();

    if s_pooled < f64::EPSILON {
        return Ok(0.0);
    }

    let d = (mean_a - mean_b) / s_pooled;
    if !d.is_finite() {
        return Err(TtsVerifyError::Dsp(DspErrorKind::Computation {
            what: "non-finite Cohen's d",
        }));
    }
    Ok(d)
}

/// Holm-Bonferroni correction for multiple comparisons.
///
/// Returns adjusted p-values. Guarantees familywise error rate control.
/// Input p-values should be raw (unadjusted) and finite.
///
/// # Errors
///
/// Returns [`TtsVerifyError::NonFiniteInput`] if any p-value is NaN or Inf.
pub fn holm_bonferroni(p_values: &[f64]) -> Result<Vec<f64>, TtsVerifyError> {
    if p_values.is_empty() {
        return Ok(Vec::new());
    }

    // Reject NaN/Inf p-values — IEEE 754 NaN comparison silently produces
    // valid-looking adjusted p-values, corrupting multiple comparison control.
    let non_finite_count = p_values.iter().filter(|p| !p.is_finite()).count();
    if non_finite_count > 0 {
        return Err(TtsVerifyError::NonFiniteInput {
            count: non_finite_count,
        });
    }

    let m = p_values.len();

    // Create (index, p_value) pairs and sort by p_value ascending
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut adjusted = vec![0.0_f64; m];
    let mut running_max = 0.0_f64;

    for (rank, &(orig_idx, p)) in indexed.iter().enumerate() {
        let multiplier = (m - rank) as f64;
        let adj = (p * multiplier).min(1.0);
        // Holm-Bonferroni enforces monotonicity: adjusted p-values must be non-decreasing
        running_max = running_max.max(adj);
        adjusted[orig_idx] = running_max;
    }

    Ok(adjusted)
}

/// Student's t CDF approximation.
///
/// Uses the regularized incomplete beta function via Abramowitz & Stegun (1964)
/// approximation 26.2.17. Accurate to ~6 significant digits.
fn student_t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 || !t.is_finite() || !df.is_finite() {
        return 0.5;
    }

    // Transform to regularized incomplete beta function:
    // I_x(a, b) where x = df/(df + t^2), a = df/2, b = 1/2
    let x = df / (df + t * t);
    let a = df / 2.0;
    let b = 0.5;

    let beta_val = regularized_incomplete_beta(x, a, b);

    if t >= 0.0 {
        1.0 - 0.5 * beta_val
    } else {
        0.5 * beta_val
    }
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction.
///
/// Uses Lentz's method for the continued fraction representation.
/// Reference: Numerical Recipes, Section 6.4.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation for better convergence:
    // I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    let threshold = (a + 1.0) / (a + b + 2.0);
    if x > threshold {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    // Log of the front factor: x^a * (1-x)^b / (a * B(a,b))
    let ln_front = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b) - a.ln();

    let front = ln_front.exp();

    // Continued fraction via Lentz's method
    let cf = beta_continued_fraction(x, a, b);

    front * cf
}

/// Continued fraction for the incomplete beta function.
///
/// Lentz's modified method. Converges rapidly for x < (a+1)/(a+b+2).
fn beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 1e-14;
    const TINY: f64 = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..=MAX_ITER {
        let m_f64 = m as f64;

        // Even step: d_{2m}
        let numer_even = m_f64 * (b - m_f64) * x / ((a + 2.0 * m_f64 - 1.0) * (a + 2.0 * m_f64));

        d = 1.0 + numer_even * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numer_even / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        result *= d * c;

        // Odd step: d_{2m+1}
        let numer_odd =
            -(a + m_f64) * (a + b + m_f64) * x / ((a + 2.0 * m_f64) * (a + 2.0 * m_f64 + 1.0));

        d = 1.0 + numer_odd * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numer_odd / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }

    result
}

/// Log of the beta function: ln(B(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Stirling's approximation for ln(Gamma(x)).
///
/// Uses Lanczos approximation for |x| > 0. Accurate to ~15 digits.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients for g=7
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Gamma(1-x)*Gamma(x) = pi/sin(pi*x)
        let reflect = std::f64::consts::PI / (std::f64::consts::PI * x).sin();
        return reflect.abs().ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = COEFFS[0];
    for (i, &coeff) in COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (x + i as f64);
    }

    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Linear interpolation percentile.
///
/// Sorts input data internally (safe for unsorted input).
/// Returns 0.0 for empty data. Percentile `p` is in range [0, 100].
pub(crate) fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi || hi >= sorted.len() {
        sorted[lo.min(sorted.len() - 1)]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// NaN-propagating maximum over an iterator of `f64`.
///
/// `f64::max` uses IEEE 754 `maxNum` semantics: `max(NaN, x) = x`, which
/// silently skips NaN elements. This function propagates NaN: if any element
/// is NaN the result is NaN, matching `fmax` behavior in many numerical
/// libraries. Returns `init` when the iterator is empty.
pub(crate) fn fold_max_propagate_nan(iter: impl Iterator<Item = f64>, init: f64) -> f64 {
    let mut acc = init;
    for v in iter {
        if v.is_nan() {
            return f64::NAN;
        }
        acc = acc.max(v);
    }
    acc
}

/// NaN-propagating minimum over an iterator of `f64`.
///
/// `f64::min` uses IEEE 754 `minNum` semantics: `min(NaN, x) = x`, which
/// silently skips NaN elements. This function propagates NaN: if any element
/// is NaN the result is NaN. Returns `init` when the iterator is empty.
pub(crate) fn fold_min_propagate_nan(iter: impl Iterator<Item = f64>, init: f64) -> f64 {
    let mut acc = init;
    for v in iter {
        if v.is_nan() {
            return f64::NAN;
        }
        acc = acc.min(v);
    }
    acc
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "stats_kani.rs"]
mod kani_proofs;

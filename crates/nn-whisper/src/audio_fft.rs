// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! FFT primitives for Whisper audio preprocessing.
//!
//! Contains Cooley-Tukey radix-2 FFT and Bluestein chirp-Z algorithm
//! for arbitrary-length DFT. Extracted from `audio.rs` for code-health
//! (500-line limit).

use std::f64::consts::PI;

use crate::WhisperError;
use nn_core::Result;
use nn_core::TensorError;

// -- Hann window (delegated to nn-core) --------------------------------------

pub(super) use nn_core::audio::hann_window;

// -- FFT (Cooley-Tukey radix-2) -----------------------------------------------

/// Next power of 2 >= n.
pub(super) fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place Cooley-Tukey radix-2 FFT on complex data `(re, im)`.
///
/// `n` must be a power of 2 and both `re` and `im` must have length >= n.
/// Uses iterative decimation-in-time with bit-reversal permutation.
/// O(n log n) per call.
///
/// Returns an error if `n` is not a power of two or buffers are too short.
pub(super) fn fft_in_place(re: &mut [f64], im: &mut [f64], n: usize) -> Result<()> {
    if !n.is_power_of_two() || re.len() < n || im.len() < n {
        return Err(WhisperError::AudioFormat {
            reason: format!(
                "fft_in_place: n must be power of 2 ({n}) with buffers len >= n (re={}, im={})",
                re.len(),
                im.len()
            ),
        }
        .into());
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterfly passes.
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let angle = angle_step * k as f64;
                let (wr, wi) = (angle.cos(), angle.sin());
                let (i1, i2) = (start + k, start + k + half);
                let tr = wr * re[i2] - wi * im[i2];
                let ti = wr * im[i2] + wi * re[i2];
                re[i2] = re[i1] - tr;
                im[i2] = im[i1] - ti;
                re[i1] += tr;
                im[i1] += ti;
            }
        }
        len <<= 1;
    }
    Ok(())
}

// -- Bluestein chirp-Z algorithm ----------------------------------------------

/// Compute arbitrary-length DFT via Bluestein's chirp-Z algorithm.
///
/// Converts an N-point DFT into circular convolutions of size M (next power
/// of 2 >= 2N-1), computed using radix-2 FFTs. O(N log N) for any N.
///
/// Writes the first `out_bins` DFT bins into `out_re` and `out_im`.
#[allow(clippy::too_many_arguments)]
pub(super) fn bluestein_dft(
    x_re: &[f64],
    x_im: &[f64],
    n: usize,
    out_re: &mut [f64],
    out_im: &mut [f64],
    out_bins: usize,
    // Pre-allocated chirp and scratch buffers (size m each).
    chirp_re: &[f64],
    chirp_im: &[f64],
    a_re: &mut [f64],
    a_im: &mut [f64],
    b_re: &mut [f64],
    b_im: &mut [f64],
    m: usize,
) -> Result<()> {
    // a[k] = x[k] * conj(chirp[k]) for k < n, zero otherwise.
    for k in 0..n {
        // conj(chirp) = (chirp_re, -chirp_im)
        a_re[k] = x_re[k] * chirp_re[k] + x_im[k] * chirp_im[k];
        a_im[k] = x_im[k] * chirp_re[k] - x_re[k] * chirp_im[k];
    }
    for k in n..m {
        a_re[k] = 0.0;
        a_im[k] = 0.0;
    }

    // b[0..n] = chirp[0..n], b[m-n+1..m] = chirp[n-1..1] (wrap-around).
    b_re[0] = chirp_re[0];
    b_im[0] = chirp_im[0];
    for k in 1..n {
        b_re[k] = chirp_re[k];
        b_im[k] = chirp_im[k];
        b_re[m - k] = chirp_re[k];
        b_im[m - k] = chirp_im[k];
    }
    for k in n..=m - n {
        b_re[k] = 0.0;
        b_im[k] = 0.0;
    }

    // FFT(a), FFT(b), pointwise multiply, IFFT.
    fft_in_place(a_re, a_im, m)?;
    fft_in_place(b_re, b_im, m)?;

    // Pointwise complex multiply.
    for k in 0..m {
        let tr = a_re[k] * b_re[k] - a_im[k] * b_im[k];
        let ti = a_re[k] * b_im[k] + a_im[k] * b_re[k];
        a_re[k] = tr;
        a_im[k] = ti;
    }

    // IFFT = conj(FFT(conj(x))) / m.
    for v in a_im.iter_mut() {
        *v = -*v;
    }
    fft_in_place(a_re, a_im, m)?;
    let inv = 1.0 / m as f64;
    for k in 0..m {
        a_re[k] *= inv;
        a_im[k] = -a_im[k] * inv;
    }

    // out[k] = conj(chirp[k]) * conv_result[k].
    for k in 0..out_bins {
        out_re[k] = a_re[k] * chirp_re[k] + a_im[k] * chirp_im[k];
        out_im[k] = a_im[k] * chirp_re[k] - a_re[k] * chirp_im[k];
    }
    Ok(())
}

// -- STFT (Bluestein FFT) ----------------------------------------------------

/// Compute power spectrogram via Bluestein's chirp-Z algorithm.
///
/// Returns flat `[n_frames, n_freqs]` row-major array where each element is
/// `re² + im²`. Handles arbitrary `n_fft` (not just powers of 2) with
/// O(n_fft × log(n_fft)) performance per frame.
pub(super) fn power_spectrogram(
    padded: &[f64],
    n_fft: usize,
    hop_length: usize,
) -> Result<Vec<f64>> {
    if n_fft == 0 || hop_length == 0 {
        return Err(WhisperError::AudioFormat {
            reason: format!(
                "power_spectrogram: n_fft ({n_fft}) and hop_length ({hop_length}) must be > 0"
            ),
        }
        .into());
    }
    if padded.len() < n_fft {
        return Err(WhisperError::AudioFormat {
            reason: format!(
                "power_spectrogram: padded length {} < n_fft {n_fft}",
                padded.len()
            ),
        }
        .into());
    }
    let n_freqs = n_fft / 2 + 1;
    let n_frames = (padded.len() - n_fft) / hop_length + 1;
    let window = hann_window(n_fft);

    // Bluestein convolution size: next power of 2 >= 2*n_fft - 1.
    // Use checked_mul to guard against usize overflow for very large n_fft.
    let bluestein_len = n_fft.checked_mul(2).ok_or_else(|| {
        TensorError::from(WhisperError::AudioFormat {
            reason: format!("power_spectrogram: 2 * n_fft ({n_fft}) overflows usize"),
        })
    })?;
    let m = next_power_of_2(bluestein_len - 1);

    // Pre-compute chirp: chirp[k] = exp(-j * π * k² / n_fft).
    let mut chirp_re = vec![0.0f64; n_fft];
    let mut chirp_im = vec![0.0f64; n_fft];
    for k in 0..n_fft {
        let angle = -PI * (k as f64 * k as f64) / n_fft as f64;
        chirp_re[k] = angle.cos();
        chirp_im[k] = angle.sin();
    }

    // Pre-allocate scratch buffers (reused across frames).
    let mut a_re = vec![0.0f64; m];
    let mut a_im = vec![0.0f64; m];
    let mut b_re = vec![0.0f64; m];
    let mut b_im = vec![0.0f64; m];
    let mut out_re = vec![0.0f64; n_freqs];
    let mut out_im = vec![0.0f64; n_freqs];

    let mut x_re = vec![0.0f64; n_fft];
    let x_im = vec![0.0f64; n_fft];

    let mut power = vec![0.0f64; n_frames * n_freqs];

    for i in 0..n_frames {
        let start = i * hop_length;

        // Apply window.
        for t in 0..n_fft {
            x_re[t] = padded[start + t] * window[t];
        }

        bluestein_dft(
            &x_re,
            &x_im,
            n_fft,
            &mut out_re,
            &mut out_im,
            n_freqs,
            &chirp_re,
            &chirp_im,
            &mut a_re,
            &mut a_im,
            &mut b_re,
            &mut b_im,
            m,
        )?;

        // Extract power.
        for k in 0..n_freqs {
            power[i * n_freqs + k] = out_re[k] * out_re[k] + out_im[k] * out_im[k];
        }
    }

    Ok(power)
}

#[cfg(kani)]
#[path = "kani_audio_fft_proofs.rs"]
mod kani_audio_fft_proofs;
